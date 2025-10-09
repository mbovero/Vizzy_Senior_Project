# vizzy/laptop/task_agent.py
# -----------------------------------------------------------------------------
# TaskAgent: one thread that owns the full user pipeline:
#   1) wait for user query (stdin for now)
#   2) wait for ScanWorker to finish if running (v1: no interrupts)
#   3) call GPT client -> get plan (kept internal to this thread)
#   4) transition to PROCESS_QUERY -> EXECUTE_TASK
#   5) execute plan via motion facade (IK later)
#   6) return to IDLE and reset idle timer
# -----------------------------------------------------------------------------

from __future__ import annotations

import threading
import time
from typing import Optional

from ..shared import config as C

from .motion import Motion
from .memory import ObjectMemory
from . import tasks       # will be added next
from .llm_task_scheduler import TaskScheduler  # NEW: LLM-based task scheduler


class TaskAgent(threading.Thread):
    def __init__(self, *, state_mgr, events):
        """
        Parameters
        ----------
        state_mgr : StateManager
            Reference to the laptop/app.py StateManager instance.
        events : Events
            The shared Events container from StateManager.
        """
        super().__init__(name="TaskAgent")
        self.state_mgr = state_mgr
        self.events = events

        # Local helpers
        self._memory = ObjectMemory(C.MEM_FILE)

        # Task scheduler: uses GPT-5 to convert user requests into structured task lists
        self._scheduler = TaskScheduler(
            model=getattr(C, "TASK_SCHEDULER_MODEL", "gpt-5")
        )

        # Motion facade: create if not already provided by StateManager
        if self.state_mgr.motion is None:
            self.state_mgr.motion = Motion(
                self.state_mgr.sock,
                self.state_mgr.mail.pwms_event,
                self.state_mgr.mail.pwms_payload,
            )
        self.motion = self.state_mgr.motion

        # Convenience aliases
        self.model = self.state_mgr.model
        self.cap = self.state_mgr.cap

        # Control
        self._stop = False

    # --------------------------------------------------------------------- API

    def stop(self) -> None:
        self._stop = True

    # --------------------------------------------------------------------- run

    def run(self) -> None:
        """
        Blocking REPL loop on stdin:
          - read a line
          - announce query_ready
          - wait for scan to finish (v1 policy: do not interrupt)
          - produce a plan via GPT
          - execute the plan
          - return to IDLE
        """
        print("[TaskAgent] Ready. Type a request (or 'help', 'exit').")

        while not self._stop:
            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_text:
                continue

            if user_text.lower() in ("exit", "quit"):
                print("[TaskAgent] Exit requested. (This does not stop the whole app.)")
                break

            if user_text.lower() in ("help", "?"):
                print("Enter a request for the arm (e.g., 'center on cup', 'move to saved mug pose').")
                continue

            # Notify orchestrator that a query exists (for state display, etc.)
            self.events.query_ready.set()

            # v1 policy: if a scan is active, wait for it to finish (no interrupt)
            if self.events.scan_active.is_set():
                print("[TaskAgent] Waiting for current search cycle to finish...")
                # Wait with small ticks so we remain responsive
                while self.events.scan_active.is_set() and not self._stop:
                    time.sleep(0.05)
            
            # Wait for LLM enrichment to complete
            if self.state_mgr.llm_worker:
                pending = self.state_mgr.llm_worker.get_pending_count()
                if pending > 0:
                    print(f"[TaskAgent] Waiting for {pending} LLM enrichment task(s) to complete...")
                    # Wait up to 30 seconds for enrichment (configurable)
                    timeout = getattr(C, "LLM_COMPLETION_TIMEOUT", 30.0)
                    completed = self.state_mgr.llm_worker.wait_for_completion(timeout=timeout)
                    if not completed:
                        print("[TaskAgent] Warning: Some enrichment tasks still pending, proceeding anyway...")

            # Transition to PROCESS_QUERY (informational; StateManager reads state for UI/telemetry)
            self.state_mgr.state = "PROCESS_QUERY"

            # Produce a plan using the task scheduler (kept internal to TaskAgent)
            try:
                # Reload memory to get latest state (including LLM-enriched semantics)
                self._memory.load()
                plan = self._scheduler.plan(user_text, self._memory)
                
                if not plan:
                    print("[TaskAgent] No tasks generated from request")
                    self.state_mgr.state = "IDLE"
                    self._reset_idle_timer()
                    self.events.query_ready.clear()
                    continue
                    
            except Exception as e:
                print(f"[TaskAgent] Planning failed: {e}")
                # Return to IDLE and reset idle timer so auto-search can resume later
                self.state_mgr.state = "IDLE"
                self._reset_idle_timer()
                self.events.query_ready.clear()
                continue

            # Transition to EXECUTE_TASK and carry out the plan
            self.state_mgr.state = "EXECUTE_TASK"
            try:
                tasks.execute_plan(
                    plan=plan,
                    motion=self.motion,
                    memory=self._memory,
                    model=self.model,
                    camera=self.cap,
                    config=C,
                    frame_sink=self.state_mgr.frame_bus.publish,
                )
            except Exception as e:
                print(f"[TaskAgent] Execution error: {e}")

            # Done: back to IDLE, reset idle timer, clear query flag
            self.state_mgr.state = "IDLE"
            self._reset_idle_timer()
            self.events.query_ready.clear()

        print("[TaskAgent] Stopped.")

    # ----------------------------------------------------------------- helpers

    def _reset_idle_timer(self) -> None:
        """After handling a query, restart the IDLE timeout so SEARCH can auto-trigger later."""
        self.state_mgr.idle_deadline = time.time() + getattr(C, "IDLE_TIMEOUT_S", 45.0)
