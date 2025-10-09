import os
import json
import uuid
import time
import threading
import queue
import concurrent.futures as cf
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# ---------------------------
# Config / Globals
# ---------------------------
load_dotenv()

IMAGE_PROCESS_MODEL = "gpt-5"
DB_PATH = "vizzy_db.json"

# Single shared lock for ALL writers (LLM workers + scan cycle)
save_db_lock = threading.Lock()


# ---------------------------
# JSON store (atomic, lock-guarded)
# ---------------------------
class JSONStore:
    def __init__(self, path: str, lock: threading.Lock):
        self.path = path
        self.lock = lock
        self._ensure_file()

    def _ensure_file(self):
        with self.lock:
            if not os.path.exists(self.path):
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump({"llm_results": {}, "scan": {}}, f, indent=2)

    def _read_unlocked(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Corrupt or empty file; return minimal structure.
            return {"llm_results": {}, "scan": {}}

    def _atomic_write_unlocked(self, data: Dict[str, Any]):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)  # atomic on POSIX & Windows

    def update(self, mutator):
        """mutator: fn(dict)->None (modify in-place) or new dict returned"""
        with self.lock:
            data = self._read_unlocked()
            out = mutator(data) or data
            self._atomic_write_unlocked(out)
            return out

    def save_llm_result(self, uid: str, payload: Dict[str, Any]):
        def _mut(d):
            d.setdefault("llm_results", {})
            d["llm_results"][uid] = payload
            return d
        return self.update(_mut)

    def save_scan_record(self, key: str, payload: Dict[str, Any]):
        def _mut(d):
            d.setdefault("scan", {})
            d["scan"][key] = payload
            return d
        return self.update(_mut)


# ---------------------------
# OpenAI helpers
# ---------------------------
def upload_image(client: OpenAI, image_path: str) -> str:
    print(f"[upload] -> {image_path}")
    with open(image_path, "rb") as image_file:
        result = client.files.create(file=image_file, purpose="vision")
    print(f"[upload] <- success: {image_path}")
    return result.id


def call_llm_for_image(file_id: str, prompt: str, model: str) -> str:
    """Do the Responses API call and return text output."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # thread-local client
    start = time.perf_counter()
    try:
        resp = client.responses.create(
            model=model,
            text={"verbosity": "low"},
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "file_id": file_id, "detail": "low"},
                    ],
                }
            ],
        )
        # Prefer output_text if available for simplicity:
        output = getattr(resp, "output_text", None)
        if not output:
            # Fallback: try to walk response structure
            # (structure can vary by SDK version)
            try:
                output = resp.output[1].content[0].text  # your original path
            except Exception:
                # As a last resort, stringify response
                output = str(resp)
        return output
    except OpenAIError as e:
        raise
    finally:
        dur = time.perf_counter() - start
        print(f"[llm] processed in {dur:.2f}s")


# ---------------------------
# Task & Worker Manager
# ---------------------------
@dataclass
class LLMTask:
    uid: str
    file_id: str
    prompt: str
    metadata: Dict[str, Any]


class WorkerManager:
    """
    Manager thread owns a ThreadPoolExecutor(max_workers=5) and a task queue.
    You can submit() tasks anytime; 5 workers stay ready for LLM calls.
    All DB writes (including scan cycle) share the same JSONStore/lock.
    """

    def __init__(
        self,
        store: JSONStore,
        max_workers: int = 5,
        model: str = IMAGE_PROCESS_MODEL,
    ):
        self.store = store
        self.model = model
        self._queue: "queue.Queue[LLMTask]" = queue.Queue()
        self._executor = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-worker")
        self._pending: Dict[str, cf.Future] = {}
        self._mgr_thread = threading.Thread(target=self._run, name="worker-manager", daemon=True)
        self._stop = threading.Event()

    def start(self):
        self._mgr_thread.start()

    def stop(self, wait: bool = True):
        # Signal manager to stop after queue drains
        self._stop.set()
        if wait:
            self._mgr_thread.join()
        self._executor.shutdown(wait=wait)

    def submit(self, file_id: str, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, cf.Future]:
        uid = str(uuid.uuid4())
        fut = cf.Future()
        self._pending[uid] = fut
        self._queue.put(LLMTask(uid=uid, file_id=file_id, prompt=prompt, metadata=metadata or {}))
        return uid, fut

    def _run(self):
        print("[manager] started")
        while not self._stop.is_set() or not self._queue.empty():
            try:
                task = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            # Submit actual work to the pool
            efut = self._executor.submit(self._do_task, task)

            # When a worker finishes, persist to DB and complete external future
            def _done(cb_fut: cf.Future, uid=task.uid):
                ext_fut = self._pending.pop(uid, None)
                try:
                    result_payload = cb_fut.result()
                    # Persist result with lock; include metadata
                    self.store.save_llm_result(uid, result_payload)
                    if ext_fut and not ext_fut.done():
                        ext_fut.set_result(result_payload)
                except Exception as e:
                    if ext_fut and not ext_fut.done():
                        ext_fut.set_exception(e)

            efut.add_done_callback(_done)
        print("[manager] stopped")

    def _do_task(self, task: LLMTask) -> Dict[str, Any]:
        # LLM call with basic retry for transient errors
        attempts = 0
        backoff = 2.0
        while True:
            attempts += 1
            try:
                text = call_llm_for_image(task.file_id, task.prompt, self.model)
                break
            except OpenAIError as e:
                if attempts >= 3:
                    raise
                time.sleep(backoff)
                backoff *= 2

        result = {
            "uid": task.uid,
            "file_id": task.file_id,
            "metadata": task.metadata,
            "model": self.model,
            "result_text": text,
            "timestamp": time.time(),
        }
        return result


# ---------------------------
# Example scan cycle writer (uses the same lock/store)
# ---------------------------
def scan_cycle_writer(store: JSONStore, iterations: int = 5, interval_s: float = 1.0):
    """
    Simulated scan cycle that writes to the same JSON file asynchronously.
    Replace this with your real scan code.
    """
    for i in range(iterations):
        key = f"scan_{int(time.time())}_{i}"
        payload = {
            "note": "scan cycle update",
            "i": i,
            "ts": time.time(),
        }
        store.save_scan_record(key, payload)
        print(f"[scan] wrote {key}")
        time.sleep(interval_s)


# ---------------------------
# Example usage
# ---------------------------
PROMPT = """
Your job is identify the center most image and nothing else.
Always finish your output. Never return partial answers.
Describe the following attributes in a couple words or 1 sentence:
1. Name of the object
2. Material type of the object
3. Color of the object
4. Any unique attributes of the object
5. Where would the most optimal pickup point be for a robotic claw.
6. Provide x and y coordinates (relative to image resolution) for the pickup point.
""".strip()


def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    store = JSONStore(DB_PATH, save_db_lock)

    # Create & start the manager (spins up the 5-thread pool under the hood)
    mgr = WorkerManager(store=store, max_workers=5, model=IMAGE_PROCESS_MODEL)
    mgr.start()

    # Upload images (same as your original helper)
    file_path1 = "white mug.png"
    file_path2 = "glass bottle.jpg"
    file_path3 = "two_bottles.jpg"

    file_id1 = upload_image(client, file_path1)
    file_id2 = upload_image(client, file_path2)
    file_id3 = upload_image(client, file_path3)

    # Queue LLM jobs — you can keep adding more at any time
    jobs = []
    for p, fid in [(file_path1, file_id1), (file_path2, file_id2), (file_path3, file_id3)]:
        uid, fut = mgr.submit(fid, PROMPT, metadata={"source_path": p})
        print(f"[submit] uid={uid} for {p}")
        jobs.append((uid, fut))

    # Start scan cycle concurrently (writes to same JSON via the same lock)
    scan_thread = threading.Thread(target=scan_cycle_writer, args=(store, 5, 0.8), daemon=True)
    scan_thread.start()

    # Do other work here if you want...
    # For demo, wait for all LLM jobs to complete and print short previews.
    for uid, fut in jobs:
        try:
            payload = fut.result()  # non-blocking in real app; you could poll or add callbacks
            preview = payload["result_text"][:180].replace("\n", " ")
            print(f"[done] {uid}: {preview} ...")
        except Exception as e:
            print(f"[error] {uid}: {e}")

    # Join scan thread for demo completeness
    scan_thread.join()

    # Stop manager gracefully
    mgr.stop(wait=True)


if __name__ == "__main__":
    main()
