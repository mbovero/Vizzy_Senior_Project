# LLM Integration Documentation

## Overview

This document describes the integration of LLM-based semantic enrichment and task scheduling into the Vizzy robotic arm system.

## Architecture

### Components

1. **Unified Memory System** (`memory.py`)
   - Thread-safe JSON storage combining object tracking and semantic data
   - Supports both unique IDs (UUID-based) and YOLO class IDs
   - Structure includes spatial coordinates (x, y, z), servo positions (pwm_btm, pwm_top), and LLM semantics

2. **LLM Semantics Module** (`llm_semantics.py`)
   - Handles image upload to OpenAI
   - Calls GPT-5 vision API for semantic enrichment
   - Parses structured semantic data (name, material, color, grasp position, etc.)

3. **LLM Worker Manager** (`llm_worker.py`)
   - ThreadPool (default 5 workers) for asynchronous LLM processing
   - Non-blocking submission of enrichment tasks
   - Automatic retry logic with exponential backoff
   - Updates memory atomically when enrichment completes

4. **Task Scheduler** (`llm_task_scheduler.py`)
   - Converts natural language user requests into structured task lists
   - Uses GPT-5 to parse intent and generate commands
   - Outputs JSON task lists with command types, targets, and destinations

5. **Integration Points**
   - `app.py`: Initializes LLM worker on startup, manages lifecycle
   - `scan_worker.py`: Captures images after centering, submits to LLM pool
   - `task_agent.py`: Uses task scheduler to plan execution from user queries

## Data Flow

### Scan Cycle with LLM Enrichment

```
1. ScanWorker centers on object
2. Gets PWM positions from servos
3. Creates object entry in memory with unique ID
4. Captures centered image
5. Saves image to disk (captured_images/)
6. Uploads image to OpenAI
7. Submits enrichment task to LLM worker pool
8. Continues scanning (non-blocking)
9. LLM worker processes image asynchronously
10. Updates object's semantics field in memory when complete
```

### Task Scheduling Flow

```
1. User inputs natural language request
2. TaskAgent waits for scan to complete (if active)
3. Loads memory with all objects (including LLM semantics)
4. Calls TaskScheduler with user request + memory context
5. GPT-5 generates structured task list
6. TaskAgent executes tasks via Motion facade
7. Returns to IDLE state
```

## Memory Structure

```json
{
  "objects": {
    "0xA1B2C3D4": {
      "id": "0xA1B2C3D4",
      "cls_id": 39,
      "cls_name": "bottle",
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "pwm_btm": 1500,
      "pwm_top": 1400,
      "semantics": {
        "name": "glass water bottle",
        "material": "glass",
        "color": "clear",
        "unique_attributes": "transparent with blue cap",
        "grasp_position": "neck below cap",
        "grasp_xy": [320, 240]
      },
      "last_seen_ts": 1234567890.123,
      "updated_this_session": 1,
      "image_path": "captured_images/0xA1B2C3D4_1234567890123.jpg"
    }
  },
  "class_index": {
    "39": ["0xA1B2C3D4"]
  }
}
```

## Task Command Types

### Low-Level Commands
- **PICK**: Pick up target object and return to resting position
- **MOVE_TO**: Move arm to destination (with or without object in claw)
- **ROTATE**: Rotate end-effector in roll/yaw directions
- **RELEASE**: Open claw to drop object or just open claw

### High-Level Commands
- **PLACE_TO**: Move to destination and release object (assumes object in claw)
- **RELOCATE**: Pick target and place at destination (combines PICK + PLACE_TO)

## Task JSON Format

```json
[
  {
    "command": "PICK",
    "target": "0xA1B2C3D4",
    "parameters": {}
  },
  {
    "command": "PLACE_TO",
    "destination": [0.5, 0.3, 0.1],
    "parameters": {}
  }
]
```

Targets and destinations can be:
- Unique object IDs (e.g., `"0xA1B2C3D4"`)
- Coordinate arrays (e.g., `[x, y, z]`)

## Configuration

All LLM-related configuration is in `shared/config.py`:

```python
IMAGE_PROCESS_MODEL = "gpt-5"        # Model for semantic enrichment
TASK_SCHEDULER_MODEL = "gpt-5"       # Model for task planning
LLM_WORKERS = 5                      # Concurrent enrichment workers
IMAGE_DIR = "captured_images"        # Directory for saved images
```

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=your_api_key_here
IMAGE_PROCESS_MODEL=gpt-5      # Optional override
TASK_SCHEDULER_MODEL=gpt-5     # Optional override
```

## API Reference

### ObjectMemory

```python
# Create object with unique ID
object_id = memory.create_object(
    cls_id=39,
    cls_name="bottle",
    pwm_btm=1500,
    pwm_top=1400,
    x=0.0, y=0.0, z=0.0,
    image_path="path/to/image.jpg"
)

# Update semantics (thread-safe)
memory.update_semantics(object_id, {
    "name": "water bottle",
    "material": "glass",
    "color": "clear"
})

# Query objects
obj = memory.get_object(object_id)
objs = memory.get_objects_by_class(cls_id=39)
all_objs = memory.list_objects()
```

### WorkerManager

```python
# Initialize (done in app.py)
worker_mgr = WorkerManager(
    memory=memory,
    max_workers=5,
    model="gpt-5"
)
worker_mgr.start()

# Submit enrichment task (non-blocking)
uid, future = worker_mgr.submit(
    object_id="0xA1B2C3D4",
    file_id="file-abc123"
)

# Stop gracefully
worker_mgr.stop(wait=True)
```

### TaskScheduler

```python
# Initialize
scheduler = TaskScheduler(model="gpt-5")

# Generate task plan
tasks = scheduler.plan(
    user_request="Pick up the blue bottle and place it on the shelf",
    memory=memory
)

# Returns list of task dicts
# [{"command": "PICK", "target": "0xA1B2C3D4"}, ...]
```

## Future Improvements

### Coordinate System Migration
Currently the system uses PWM positions. Future updates will:
1. Replace `pwm_btm`, `pwm_top` with motor rotation angles for 5-axis arm
2. Calculate `x, y` from inverse kinematics
3. Read `z` from laser sensor attached to end-effector

### Additional Command Types
Consider adding:
- **INSPECT**: Move to viewing angle and capture image
- **MEASURE**: Use sensors to measure object dimensions
- **WAIT**: Pause for specified duration
- **HOME**: Return to home/resting position

### Enhanced Semantics
- Add object dimensions from vision or sensors
- Include material properties (weight, fragility)
- Track object state changes (empty/full, open/closed)

## Troubleshooting

### LLM enrichment not happening
- Check `OPENAI_API_KEY` is set in `.env`
- Verify `llm_worker` is passed to `ScanWorker` in `app.py`
- Check console for upload/enrichment errors

### Task scheduler not generating tasks
- Ensure memory is loaded with objects (run a scan first)
- Check that objects have semantics (wait for enrichment to complete)
- Verify OpenAI API quota and rate limits

### Images not being captured
- Check `IMAGE_DIR` exists and is writable
- Verify camera is accessible and returning frames
- Look for errors in `_capture_and_enrich()` output

## Testing

### Test Semantic Enrichment
```python
from software.vizzy.laptop.llm_semantics import upload_image, call_llm_for_semantics

# Upload test image
file_id = upload_image("test_image.jpg")

# Get semantics
semantics = call_llm_for_semantics(file_id)
print(semantics)
```

### Test Task Scheduler
```python
from software.vizzy.laptop.llm_task_scheduler import plan_tasks
from software.vizzy.laptop.memory import ObjectMemory

memory = ObjectMemory("object_memory.json")
tasks = plan_tasks("Pick up the red cup", memory)
print(tasks)
```

## Performance Considerations

- LLM enrichment runs asynchronously (doesn't block scanning)
- Default 5 workers balances throughput vs API rate limits
- Images saved to disk for debugging/reprocessing
- Memory updates are atomic and thread-safe
- Task planning waits for scan completion (v1 policy)

## License & Credits

Part of the Vizzy Senior Project.
Integrates OpenAI GPT-5 for vision and language understanding.

