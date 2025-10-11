# llm_worker.py
import os
import json
import time
import uuid
import queue
import threading
import concurrent.futures as cf
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

IMAGE_PROCESS_MODEL = os.getenv("IMAGE_PROCESS_MODEL", "gpt-5")



# =========================
# JSON Store (objects by ID)
# =========================
class JSONStore:
    """
    Thread-safe, atomic JSON DB with layout:
    {
      "objects": {
        "<id>": {
          "id": "<id>",
          "classification": "...",
          "x": 0.0, "y": 0.0, "z": 0.0,
          "semantics": { ... },       # filled by LLM
          ... (any other scan-cycle fields)
        }
      }
    }
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self):
        with self._lock:
            if not os.path.exists(self.path):
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump({"objects": {}}, f, indent=2)

    def _read_unlocked(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {"objects": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"objects": {}}

    def _atomic_write_unlocked(self, data: Dict[str, Any]):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)

    def update(self, mutator):
        with self._lock:
            data = self._read_unlocked()
            out = mutator(data) or data
            self._atomic_write_unlocked(out)
            return out

    # ---- object helpers ----
    # TODO Change this to use a simplified ID method. Such as a count....
    @staticmethod
    def _new_object_id() -> str:
        # Short, readable hex ID like "0xA1B2C3D4"
        return "0x" + uuid.uuid4().hex[:8].upper()

    def create_object(self,
                      classification: str,
                      x: float,
                      y: float,
                      z: float,
                      extra: Optional[Dict[str, Any]] = None,
                      object_id: Optional[str] = None) -> str:
        """Create a new object entry and return its ID."""
        oid = object_id or self._new_object_id()
        base = {
            "id": oid,
            "classification": classification,
            "x": float(x),
            "y": float(y),
            "z": float(z),
        }
        if extra:
            base.update(extra)

        def _mut(d):
            d.setdefault("objects", {})
            if oid in d["objects"]:
                raise ValueError(f"Object ID already exists: {oid}")
            d["objects"][oid] = base
            return d

        self.update(_mut)
        return oid

    def upsert_object_fields(self, object_id: str, patch: Dict[str, Any]):
        """Merge top-level fields (but do not allow 'id' overwrite)."""
        patch = dict(patch)
        patch.pop("id", None)

        def _mut(d):
            d.setdefault("objects", {})
            obj = d["objects"].setdefault(object_id, {"id": object_id})
            obj.update(patch)
            d["objects"][object_id] = obj
            return d

        self.update(_mut)

    def update_semantics(self, object_id: str, semantics: Dict[str, Any]):
        """Deep-merge 'semantics' for an existing object."""
        def _mut(d):
            objs = d.setdefault("objects", {})
            obj = objs.setdefault(object_id, {"id": object_id})
            sem = obj.setdefault("semantics", {})
            sem.update(semantics or {})
            obj["semantics"] = sem
            objs[object_id] = obj
            return d

        self.update(_mut)

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            data = self._read_unlocked()
            return data.get("objects", {}).get(object_id)

    def list_objects(self) -> List[Dict[str, Any]]:
        with self._lock:
            data = self._read_unlocked()
            return list(data.get("objects", {}).values())


# =========================
# OpenAI helpers
# =========================
def upload_image(image_path: str, *, api_key: Optional[str] = None) -> str:
    """Upload an image file to OpenAI 'vision' storage; return file_id."""
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    with open(image_path, "rb") as f:
        res = client.files.create(file=f, purpose="vision")
    return res.id


SEMANTICS_PROMPT = (
    "Identify only the center-most object. Return strictly JSON with keys: "
    '{"name","material","color","unique_attributes","grasp_position","grasp_xy"}. '
    '"grasp_xy" must be a two-element array of pixel coordinates [x, y]. '
    "No extra text—JSON only."
)


def _try_parse_semantics(text: str) -> Dict[str, Any]:
    """
    Robustly parse semantics from model output.
    Accepts raw JSON, or JSON inside code fences. Falls back to minimal mapping.
    """
    import re
    cand = text.strip()

    # Extract fenced JSON if present
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", cand)
    if m:
        cand = m.group(1)

    try:
        obj = json.loads(cand)
        # Coerce required fields & types
        out = {
            "name": str(obj.get("name", "")),
            "material": str(obj.get("material", "")),
            "color": str(obj.get("color", "")),
            "unique_attributes": str(obj.get("unique_attributes", "")),
            "grasp_position": str(obj.get("grasp_position", "")),
            "grasp_xy": obj.get("grasp_xy", [0, 0]),
        }
        # normalize grasp_xy
        gx = out["grasp_xy"]
        if isinstance(gx, (list, tuple)) and len(gx) == 2:
            try:
                out["grasp_xy"] = [int(round(float(gx[0]))), int(round(float(gx[1])))]
            except Exception:
                out["grasp_xy"] = [0, 0]
        else:
            out["grasp_xy"] = [0, 0]
        return out
    except Exception:
        # If totally unparsable, stash raw output into unique_attributes
        return {
            "name": "",
            "material": "",
            "color": "",
            "unique_attributes": f"raw: {text[:200]}",
            "grasp_position": "",
            "grasp_xy": [0, 0],
        }


def _call_llm_for_semantics(file_id: str, model: str) -> Dict[str, Any]:
    """
    Call Responses API asking for strict JSON semantics.
    Uses response_format when available; falls back gracefully.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    inputs = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": SEMANTICS_PROMPT},
            {"type": "input_image", "file_id": file_id, "detail": "low"},
        ],
    }]

    # Try JSON schema mode if supported
    try:
        resp = client.responses.create(
            model=model,
            input=inputs,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "semantics",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "material": {"type": "string"},
                            "color": {"type": "string"},
                            "unique_attributes": {"type": "string"},
                            "grasp_position": {"type": "string"},
                            "grasp_xy": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2
                            }
                        },
                        "required": ["name","material","color","unique_attributes","grasp_position","grasp_xy"],
                        "additionalProperties": False
                    }
                }
            },
        )
        raw = getattr(resp, "output_text", None) or str(resp)
    except TypeError:
        # Older SDKs without response_format
        resp = client.responses.create(model=model, input=inputs, text={"verbosity": "low"})
        raw = getattr(resp, "output_text", None)
        if not raw:
            try:
                raw = resp.output[1].content[0].text
            except Exception:
                raw = str(resp)

    return _try_parse_semantics(raw)


# =========================
# Worker Manager
# =========================
@dataclass
class LLMTask:
    uid: str
    object_id: str
    file_id: str


class WorkerManager:
    """
    Manager thread + ThreadPoolExecutor(max_workers=5).
    Each task targets an existing object_id; the worker writes 'semantics' into it.
    """

    def __init__(self, store: JSONStore, max_workers: int = 5, model: str = IMAGE_PROCESS_MODEL):
        self.store = store
        self.model = model
        self._queue: "queue.Queue[LLMTask]" = queue.Queue()
        self._executor = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-worker")
        self._pending: Dict[str, cf.Future] = {}
        self._stop = threading.Event()
        self._mgr_thread = threading.Thread(target=self._run, name="worker-manager", daemon=True)

    def start(self):
        self._mgr_thread.start()

    def stop(self, wait: bool = True):
        self._stop.set()
        if wait:
            self._mgr_thread.join()
        self._executor.shutdown(wait=wait)

    def submit(self, *, object_id: str, file_id: str) -> Tuple[str, cf.Future]:
        uid = uuid.uuid4().hex
        fut = cf.Future()
        self._pending[uid] = fut
        self._queue.put(LLMTask(uid=uid, object_id=object_id, file_id=file_id))
        return uid, fut

    # ---- internals ----
    def _run(self):
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
                    if ext_fut and not ext_fut.done():
                        ext_fut.set_exception(e)

            efut.add_done_callback(_done)

    def _do_task(self, task: LLMTask) -> Dict[str, Any]:
        t0 = time.perf_counter()
        attempts = 0
        backoff = 2.0
        while True:
            attempts += 1
            try:
                semantics = _call_llm_for_semantics(task.file_id, self.model)
                break
            except OpenAIError:
                if attempts >= 3:
                    raise
                time.sleep(backoff)
                backoff *= 2
        dt = time.perf_counter() - t0

        # Write into the correct object
        self.store.update_semantics(task.object_id, semantics)

        return {
            "uid": task.uid,
            "object_id": task.object_id,
            "file_id": task.file_id,
            "semantics": semantics,
            "model": self.model,
            "duration_s": round(dt, 3),
        }


__all__ = [
    "JSONStore",
    "WorkerManager",
    "upload_image",
    "IMAGE_PROCESS_MODEL",
]