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

        # Use shared memory instance from StateManager
        self._memory = state_mgr.memory

        # Task scheduler: uses GPT-5 to convert user requests into structured task lists
        self._scheduler = TaskScheduler(
            model=C.TASK_SCHEDULER_MODEL
        )

        # Motion facade: create if not already provided by StateManager
        if self.state_mgr.motion is None:
            self.state_mgr.motion = Motion(
                self.state_mgr.sock,
                self.state_mgr.mail.cmd_complete_q,
                self.state_mgr.mail.obj_loc_event,
                self.state_mgr.mail.obj_loc_payload,
                abort_event=self.state_mgr.events.scan_abort,
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
            
            # Wait for LLM enrichment to complete (skip if directly accessing task scheduler)
            skip_scan = C.SKIP_TO_TASK_SCHEDULER
            if not skip_scan and self.state_mgr.llm_worker:
                pending = self.state_mgr.llm_worker.get_pending_count()
                if pending > 0:
                    print(f"[TaskAgent] Waiting for {pending} LLM enrichment task(s) to complete...")
                    # Wait up to 30 seconds for enrichment (configurable)
                    timeout = C.LLM_COMPLETION_TIMEOUT
                    completed = self.state_mgr.llm_worker.wait_for_completion(timeout=timeout)
                    if not completed:
                        print("[TaskAgent] Warning: Some enrichment tasks still pending, proceeding anyway...")
            elif skip_scan:
                print("[TaskAgent] SKIP_TO_TASK_SCHEDULER: Using existing memory, skipping enrichment wait")

            # Transition to PROCESS_QUERY (informational; StateManager reads state for UI/telemetry)
            self.state_mgr.state = "PROCESS_QUERY"

            # Produce a plan using the task scheduler (kept internal to TaskAgent)
            try:
                print(f"[TaskAgent] Processing query: '{user_text}'")
                # Reload memory to get latest state (including LLM-enriched semantics)
                print("[TaskAgent] Reloading memory...")
                self._memory.load()
                print(f"[TaskAgent] Memory has {len(self._memory.list_objects())} objects")
                
                print("[TaskAgent] Calling task scheduler...")
                plan = self._scheduler.plan(user_text, self._memory)
                print(f"[TaskAgent] Scheduler returned {len(plan) if plan else 0} tasks")
                
                # Save the plan to a file for verification
                self._save_plan_to_file(user_text, plan)
                
                if not plan:
                    print("[TaskAgent] No tasks generated from request")
                    self.state_mgr.state = "IDLE"
                    self._reset_idle_timer()
                    self.events.query_ready.clear()
                    continue
                    
            except Exception as e:
                import traceback
                print(f"[TaskAgent] Planning failed with exception: {type(e).__name__}: {e}")
                print(f"[TaskAgent] Full traceback:")
                traceback.print_exc()
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
                    dispatcher=self.state_mgr.dispatcher,
                    memory=self._memory,
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
        self.state_mgr.idle_deadline = time.time() + C.IDLE_TIMEOUT_S
    
    def _save_plan_to_file(self, user_query: str, plan: list) -> None:
        """Save the task scheduler output to a file for verification."""
        import json
        from datetime import datetime
        
        output_file = C.TASK_SCHEDULER_OUTPUT_FILE
        
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "plan": plan,
            "num_tasks": len(plan) if plan else 0,
            "memory_objects": len(self._memory.list_objects())
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"[TaskAgent] Plan saved to: {output_file}")
        except Exception as e:
            print(f"[TaskAgent] Warning: Failed to save plan to file: {e}")
