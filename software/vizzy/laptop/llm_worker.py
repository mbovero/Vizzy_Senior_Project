# vizzy/laptop/llm_worker.py
# -----------------------------------------------------------------------------
# WorkerManager: ThreadPool for asynchronous LLM semantic enrichment.
# Submits image processing tasks to a pool of workers that call GPT-5 vision API.
# Results are written to the unified ObjectMemory.
# -----------------------------------------------------------------------------

import os
import time
import queue
import threading
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAIError

from .llm_semantics import call_llm_for_semantics, IMAGE_PROCESS_MODEL
from .memory import ObjectMemory

load_dotenv()


@dataclass
class LLMTask:
    """
    A task for semantic enrichment.
    
    Attributes:
        uid: Unique task identifier
        object_id: The unique object ID in memory to update
        file_id: OpenAI file ID of the uploaded image
    """
    uid: str
    object_id: str
    file_id: str


class WorkerManager:
    """
    Manager thread + ThreadPoolExecutor for LLM semantic enrichment.
    
    Each task targets an existing object_id in ObjectMemory; the worker
    calls the LLM, gets semantic data, and writes it into the object's
    'semantics' field.
    
    The scan cycle can submit tasks without blocking, and workers process
    them asynchronously.
    """

    def __init__(
        self,
        memory: ObjectMemory,
        max_workers: int = 5,
        model: str = IMAGE_PROCESS_MODEL
    ):
        """
        Initialize the worker manager.
        
        Args:
            memory: Unified ObjectMemory instance (thread-safe)
            max_workers: Number of concurrent LLM worker threads
            model: Model name for semantic enrichment
        """
        self.memory = memory
        self.model = model
        self._queue: "queue.Queue[LLMTask]" = queue.Queue()
        self._executor = cf.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="llm-worker"
        )
        self._pending: Dict[str, cf.Future] = {}
        self._stop = threading.Event()
        self._mgr_thread = threading.Thread(
            target=self._run,
            name="worker-manager",
            daemon=True
        )

    def start(self) -> None:
        """Start the manager thread (which manages the worker pool)."""
        self._mgr_thread.start()
        print(f"[LLM WorkerManager] Started with {self._executor._max_workers} workers")

    def stop(self, wait: bool = True) -> None:
        """
        Stop the manager gracefully.
        
        Args:
            wait: If True, wait for all pending tasks to complete
        """
        self._stop.set()
        if wait:
            self._mgr_thread.join()
        self._executor.shutdown(wait=wait)
        print("[LLM WorkerManager] Stopped")

    def submit(self, *, object_id: str, file_id: str) -> Tuple[str, cf.Future]:
        """
        Submit a semantic enrichment task.
        
        Args:
            object_id: The unique object ID to update
            file_id: OpenAI file ID of the uploaded image
        
        Returns:
            Tuple of (task_uid, future) for tracking completion
        """
        import uuid
        uid = uuid.uuid4().hex
        fut = cf.Future()
        self._pending[uid] = fut
        self._queue.put(LLMTask(uid=uid, object_id=object_id, file_id=file_id))
        print(f"[LLM WorkerManager] Queued task {uid[:8]} for object {object_id}")
        return uid, fut

    # ---- internals ----
    
    def _run(self) -> None:
        """Manager loop: dequeue tasks and submit to thread pool."""
        while not self._stop.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            efut = self._executor.submit(self._do_task, task)

            def _done(cb_fut: cf.Future, uid=task.uid):
                ext_fut = self._pending.pop(uid, None)
                try:
                    payload = cb_fut.result()
                    if ext_fut and not ext_fut.done():
                        ext_fut.set_result(payload)
                except Exception as e:
                    print(f"[LLM WorkerManager] Task {uid[:8]} failed: {e}")
                    if ext_fut and not ext_fut.done():
                        ext_fut.set_exception(e)

            efut.add_done_callback(_done)

    def _do_task(self, task: LLMTask) -> Dict[str, Any]:
        """
        Execute a single semantic enrichment task with retry logic.
        
        Args:
            task: The LLMTask to process
        
        Returns:
            Dictionary with task results and metadata
        """
        t0 = time.perf_counter()
        attempts = 0
        backoff = 2.0
        
        # Retry logic for transient API errors
        while True:
            attempts += 1
            try:
                semantics = call_llm_for_semantics(task.file_id, self.model)
                break
            except OpenAIError as e:
                if attempts >= 3:
                    print(f"[LLM Worker] Failed after {attempts} attempts: {e}")
                    # Return empty semantics on failure
                    semantics = {
                        "name": "",
                        "material": "",
                        "color": "",
                        "unique_attributes": f"LLM error: {str(e)[:100]}",
                        "grasp_position": "",
                        "grasp_xy": [0, 0],
                    }
                    break
                print(f"[LLM Worker] Retry {attempts}/3 after error: {e}")
                time.sleep(backoff)
                backoff *= 2
        
        dt = time.perf_counter() - t0

        # Write semantics into the object in memory
        self.memory.update_semantics(task.object_id, semantics)
        
        print(f"[LLM Worker] Enriched object {task.object_id} in {dt:.2f}s")

        return {
            "uid": task.uid,
            "object_id": task.object_id,
            "file_id": task.file_id,
            "semantics": semantics,
            "model": self.model,
            "duration_s": round(dt, 3),
        }


__all__ = [
    "WorkerManager",
    "LLMTask",
]

