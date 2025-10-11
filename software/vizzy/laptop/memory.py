from __future__ import annotations
import json, os, time, threading, uuid
from typing import Dict, Any, List, Optional

# TODO: remove pwm positions and replace with object XYZ coordinates and claw grasping orientation; 
# also remove avg_conf as I don't see us using this in the future
class ObjectMemory:
    """
    Unified persistent memory combining JSONStore and ObjectMemory functionality.
    Thread-safe with atomic writes.
    
    Structure:
    {
      "objects": {
        "<unique_id>": {
          "id": "<unique_id>",           # Unique UUID-based ID (e.g., "0xA1B2C3D4")
          "cls_id": int,                 # YOLO class ID (for backward compatibility)
          "cls_name": str,               # Human-readable class name
          "x": float,                    # X coordinate (from IK)
          "y": float,                    # Y coordinate (from IK)
          "z": float,                    # Z coordinate (laser sensor height)
          "pwm_btm": int,                # Bottom servo PWM (to be removed later)
          "pwm_top": int,                # Top servo PWM (to be removed later)
          "semantics": {...},            # LLM-enriched semantic data
          "last_seen_ts": float,         # UNIX timestamp
          "updated_this_session": int,   # 1 if updated in current session, else 0
          "image_path": str,             # Optional: path to captured image
        }
      },
      "class_index": {
        "<cls_id>": ["<unique_id1>", "<unique_id2>", ...]  # Index for quick class lookup
      }
    }
    """

    def __init__(self, path: str):
        """
        Create an ObjectMemory instance and immediately load existing data
        from the given file path if it exists.
        """
        self.path = path
        self.data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        """
        Load JSON data from disk into memory.
        If the file does not exist or fails to parse, start with empty data.
        """
        with self._lock:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        # Ensure proper structure
                        if isinstance(loaded, dict) and "objects" in loaded:
                            self.data = loaded
                        else:
                            # Legacy format migration
                            self.data = {"objects": {}, "class_index": {}}
                else:
                    self.data = {"objects": {}, "class_index": {}}
            except Exception as e:
                print(f"[Memory] Failed to load {self.path}: {e}")
                self.data = {"objects": {}, "class_index": {}}

    def _save_unlocked(self) -> None:
        """
        Internal save method that does NOT acquire the lock.
        Should only be called when lock is already held.
        """
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except Exception as e:
            print(f"[Memory] Failed to save {self.path}: {e}")

    def save(self) -> None:
        """
        Save current data to disk atomically:
        - Write to a temporary file first
        - Replace the original file to avoid partial writes
        """
        with self._lock:
            self._save_unlocked()

    @staticmethod
    def _new_object_id() -> str:
        """Generate a short, readable hex ID like '0xA1B2C3D4'"""
        return "0x" + uuid.uuid4().hex[:8].upper()

    def reset_session_flags(self) -> None:
        """
        Reset the 'updated_this_session' flag for all entries to 0.
        This should be called at the start of a new search session.
        """
        with self._lock:
            for obj in self.data.get("objects", {}).values():
                obj["updated_this_session"] = 0
            self._save_unlocked()  # Use unlocked version since we hold the lock

    def prune_not_updated(self) -> None:
        """
        Remove any entries that were not updated in the current session
        (i.e., 'updated_this_session' != 1).
        Also rebuild the class_index.
        """
        with self._lock:
            objects = self.data.get("objects", {})
            before = len(objects)
            
            # Filter out non-updated objects
            updated_objects = {
                oid: obj for oid, obj in objects.items()
                if int(obj.get("updated_this_session", 0)) == 1
            }
            
            self.data["objects"] = updated_objects
            after = len(updated_objects)
            
            if before != after:
                print(f"[Memory] Pruned {before - after} stale entries")
            
            # Rebuild class index
            self._rebuild_class_index()
            self._save_unlocked()  # Use unlocked version since we hold the lock

    def _rebuild_class_index(self) -> None:
        """Rebuild the class_index from current objects."""
        class_index: Dict[str, List[str]] = {}
        for oid, obj in self.data.get("objects", {}).items():
            cls_id_str = str(obj.get("cls_id", ""))
            if cls_id_str:
                class_index.setdefault(cls_id_str, []).append(oid)
        self.data["class_index"] = class_index

    def create_object(
        self,
        cls_id: int,
        cls_name: str,
        pwm_btm: int,
        pwm_top: int,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        image_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new object entry with a unique ID.
        Returns the generated unique ID.
        
        Args:
            cls_id: YOLO class ID
            cls_name: Human-readable name
            pwm_btm: Bottom servo PWM position
            pwm_top: Top servo PWM position
            x, y, z: Spatial coordinates
        
        Returns:
            The unique object ID (e.g., "0xA1B2C3D4")
        """
        with self._lock:
            oid = self._new_object_id()
            now = time.time()
            
            obj = {
                "id": oid,
                "cls_id": int(cls_id),
                "cls_name": cls_name,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "pwm_btm": int(pwm_btm),
                "pwm_top": int(pwm_top),
                "semantics": {},
                "last_seen_ts": now,
                "updated_this_session": 1,
            }
            
            if image_path:
                obj["image_path"] = image_path
            
            if extra:
                obj.update(extra)
            
            # Store object
            self.data.setdefault("objects", {})[oid] = obj
            
            # Update class index
            cls_id_str = str(cls_id)
            self.data.setdefault("class_index", {}).setdefault(cls_id_str, []).append(oid)
            
            self._save_unlocked()  # Use unlocked version since we hold the lock
            return oid

    def update_entry(
        self,
        cls_id: int,
        cls_name: str,
        pwm_btm: int,
        pwm_top: int,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        image_path: Optional[str] = None,
        orientation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new object entry (backward compatible with old API).
        Always creates a new unique object since we may have duplicate class_ids.
        
        Returns the generated unique ID.
        """
        extra = {}
        if orientation:
            extra["orientation"] = orientation
        
        return self.create_object(
            cls_id=cls_id,
            cls_name=cls_name,
            pwm_btm=pwm_btm,
            pwm_top=pwm_top,
            x=x,
            y=y,
            z=z,
            image_path=image_path,
            extra=extra if extra else None,
        )

    def update_semantics(self, object_id: str, semantics: Dict[str, Any]) -> None:
        """
        Update semantic information for an existing object.
        Deep-merges with existing semantics.
        
        Args:
            object_id: The unique object ID
            semantics: Dictionary of semantic attributes from LLM
        """
        with self._lock:
            objects = self.data.setdefault("objects", {})
            if object_id in objects:
                obj = objects[object_id]
                sem = obj.setdefault("semantics", {})
                sem.update(semantics or {})
                obj["semantics"] = sem
                objects[object_id] = obj
                self._save_unlocked()  # Use unlocked version since we hold the lock
            else:
                print(f"[Memory] Warning: Cannot update semantics for unknown object {object_id}")

    def get_object(self, object_id: str) -> Optional[Dict[str, Any]]:
        """Get a single object by its unique ID."""
        with self._lock:
            return self.data.get("objects", {}).get(object_id)

    def get_objects_by_class(self, cls_id: int) -> List[Dict[str, Any]]:
        """Get all objects with a specific YOLO class_id."""
        with self._lock:
            cls_id_str = str(cls_id)
            oids = self.data.get("class_index", {}).get(cls_id_str, [])
            objects = self.data.get("objects", {})
            return [objects[oid] for oid in oids if oid in objects]

    def list_objects(self) -> List[Dict[str, Any]]:
        """Return all objects as a list."""
        with self._lock:
            return list(self.data.get("objects", {}).values())

    def entries_sorted(self) -> List[dict]:
        """
        Return all entries sorted by integer class_id (for backward compatibility).
        """
        with self._lock:
            objects = self.data.get("objects", {})
            return sorted(objects.values(), key=lambda obj: int(obj.get("cls_id", 0)))
