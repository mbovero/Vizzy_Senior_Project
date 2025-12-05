# vizzy/laptop/llm_task_scheduler.py
# -----------------------------------------------------------------------------
# TaskScheduler: Uses GPT-5 to parse user requests into structured task lists.
# Takes user query + memory context -> returns list of task commands.
# -----------------------------------------------------------------------------

import os
import json
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .memory import ObjectMemory

load_dotenv()

TASK_SCHEDULER_MODEL = os.getenv("TASK_SCHEDULER_MODEL", "gpt-5")


# Task command types
# Note: LLM outputs simple command names (e.g., "PICK" not "CMD_PICK")
COMMANDS = {
    # High-level commands (expanded by laptop into primitives)
    "PICK": "Pick up target object and return to rest position (expanded by system)",
    "PLACE": "Place object at destination with optional offset (expanded by system)",
    
    # Low-level primitive commands (sent directly to RPi)
    "GRAB": "Close claw/gripper",
    "RELEASE": "Open claw/gripper",
    "MOVE_TO": "Move end-effector to XYZ position (millimeters) with optional offset",
    "ROT_YAW": "Rotate yaw axis by specified angle (degrees)",
    "ROT_PITCH": "Rotate pitch axis by specified angle (degrees)",
}


TASK_SCHEDULER_PROMPT = """You are a task planning system for a 5-axis robotic arm with a gripper.

The user will provide a natural language request. You must parse it into a structured list of tasks.
However, do not parse any harmful request such as stab someone or something along those lines...

IMPORTANT: All coordinates are in MILLIMETERS.

Coordinate system (robot frame) directions:
- Positive Y (y > 0) is "right"; negative Y (y < 0) is "left".
- Positive X (x > 0) is "forward"; negative X (x < 0) is "backward".

Available command types:

HIGH-LEVEL COMMANDS (automatically expanded by the system):
1. PICK
   - Picks up an object and returns to rest position
   - System expands to: RELEASE → MOVE_TO → ROT_YAW → GRAB → MOVE_TO(rest) → ROT_YAW(rest)
   - Required field: "target" (object ID or [x_mm, y_mm, z_mm])
   - Example: {{"command": "PICK", "target": "0xA1B2C3D4"}}

2. PLACE
   - Places held object at destination and returns to rest position
   - System expands to: MOVE_TO → RELEASE → MOVE_TO(rest) → ROT_YAW(rest)
   - Required field: "destination" (object ID or [x_mm, y_mm, z_mm])
   - Optional field: "offset" ([x_mm, y_mm, z_mm] - relative displacement)
   - Example: {{"command": "PLACE", "destination": [1200, 800, 500], "offset": [0, 0, 50]}}

LOW-LEVEL PRIMITIVES (use only when high-level commands don't apply):
3. GRAB - Close gripper
   - Example: {{"command": "GRAB"}}

4. RELEASE - Open gripper
   - Example: {{"command": "RELEASE"}}

5. MOVE_TO - Move to XYZ position (millimeters)
   - Required field: "destination" (object ID or [x_mm, y_mm, z_mm])
   - Optional field: "offset" ([x_mm, y_mm, z_mm])
   - Optional field: "pitch" (degrees) - specify pitch angle for intelligent grasping
   - Use MOVE_TO with pitch specified for intelligent positioning with proper claw orientation
   - Example: {{"command": "MOVE_TO", "destination": [1200, 800, 500], "pitch": -90.0}}
   - Example: {{"command": "MOVE_TO", "destination": [1200, 800, 500]}}  # Uses current pitch/yaw

6. ROT_YAW - Rotate yaw axis (servo only, no arm movement)
   - Rotates only the yaw servo motor without moving the arm
   - Keeps current arm position (x, y, z) and pitch unchanged
   - Use for "arbitrary" claw rotations at a fixed position
   - For intelligent grasping with specific pitch, use MOVE_TO with pitch specified instead
   - Required field: "angle" (degrees)
   - Example: {{"command": "ROT_YAW", "angle": 45.0}}

7. ROT_PITCH - Rotate pitch axis (servo only, no arm movement)
   - Rotates only the pitch servo motor without moving the arm
   - Keeps current arm position (x, y, z) and yaw unchanged
   - Use for "arbitrary" claw rotations at a fixed position
   - For intelligent grasping with specific pitch, use MOVE_TO with pitch specified instead
   - Required field: "angle" (degrees)
   - Example: {{"command": "ROT_PITCH", "angle": 30.0}}

CRITICAL RULES:
- Use object IDs (like "0xA1B2C3D4") when referring to known objects from memory
- Use coordinate arrays [x, y, z] for absolute positions (in millimeters!)
- For picking and placing: use PICK and PLACE commands (they're easier and automatic)
- Only use primitives when you need fine control
- Offset is always optional and relative (e.g., [0, 0, 50] means 50mm above)
- Command names must be exact: "PICK", "PLACE", "GRAB", "RELEASE", "MOVE_TO", "ROT_YAW", "ROT_PITCH"
- For intelligent grasping: Use MOVE_TO with pitch specified to approach objects with proper claw orientation
- For arbitrary rotations: Use ROT_YAW/ROT_PITCH only when you need to adjust claw orientation at a fixed position (servo-only, no arm movement)
- ROT_YAW and ROT_PITCH are for fine-tuning claw orientation, not for intelligent positioning
- Additional information is that if the user asks for a wave, that the pitch should be set to 0 degrees to accomplished that goal
Memory context (objects currently in the workspace):
{memory_context}

User request: {user_request}

Return ONLY a JSON array of task objects. No markdown, no extra text, just the JSON array.
Example format: [{{"command": "PICK", "target": "0xA1B2C3D4"}}, {{"command": "PLACE", "destination": [1200, 800, 500]}}]
"""


def _build_memory_context(memory: ObjectMemory) -> str:
    """
    Build a human-readable summary of objects in memory for the LLM prompt.
    
    Args:
        memory: ObjectMemory instance
    
    Returns:
        Formatted string with object details
    """
    objects = memory.list_objects()
    if not objects:
        return "No objects currently in memory."
    
    lines = []
    for obj in objects:
        oid = obj.get("id", "")
        cls_name = obj.get("cls_name", "unknown")
        x, y, z = obj.get("x", 0), obj.get("y", 0), obj.get("z", 0)
        semantics = obj.get("semantics", {})
        
        desc = f"- Object {oid}: {cls_name}"
        if semantics:
            name = semantics.get("name", "")
            color = semantics.get("color", "")
            material = semantics.get("material", "")
            if name:
                desc += f" ({name})"
            if color:
                desc += f", color={color}"
            if material:
                desc += f", material={material}"
        desc += f", position=[{x:.2f}, {y:.2f}, {z:.2f}]"
        lines.append(desc)
    
    return "\n".join(lines)


def _parse_task_list(text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM output into a list of task dictionaries.
    Handles JSON arrays in raw form or within code fences.
    
    Args:
        text: Raw output from the LLM
    
    Returns:
        List of task dictionaries
    """
    cand = text.strip()
    
    print(f"[TaskScheduler] Raw LLM output:\n{text}")
    
    # Try to extract JSON from code fences
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", cand)
    if m:
        cand = m.group(1)
        print(f"[TaskScheduler] Extracted from code fence")
    
    # Also try to find JSON array without code fences
    if not m:
        array_match = re.search(r'\[[\s\S]*\]', cand)
        if array_match:
            cand = array_match.group(0)
            print(f"[TaskScheduler] Found JSON array")
    
    print(f"[TaskScheduler] Attempting to parse: {cand[:200]}...")
    
    try:
        tasks = json.loads(cand)
        if not isinstance(tasks, list):
            print(f"[TaskScheduler] Warning: Expected list, got {type(tasks)}")
            return []
        
        # Validate each task has required fields
        valid_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                print(f"[TaskScheduler] Warning: Task {i} is not a dict, skipping")
                continue
            
            cmd = task.get("command", "").upper()
            if cmd not in COMMANDS:
                print(f"[TaskScheduler] Warning: Unknown command '{cmd}' in task {i}")
                continue
            
            # Normalize command to uppercase
            task["command"] = cmd
            valid_tasks.append(task)
        
        return valid_tasks
    
    except json.JSONDecodeError as e:
        print(f"[TaskScheduler] JSON parsing error: {e}")
        print(f"[TaskScheduler] Failed at position {e.pos}")
        print(f"[TaskScheduler] Near: {cand[max(0, e.pos-50):min(len(cand), e.pos+50)]}")
        print(f"[TaskScheduler] Full output length: {len(text)} chars")
        return []


def plan_tasks(user_request: str, memory: ObjectMemory, model: str = TASK_SCHEDULER_MODEL) -> List[Dict[str, Any]]:
    """
    Generate a task plan from a user request using GPT-5.
    
    Args:
        user_request: Natural language request from user
        memory: ObjectMemory with current workspace state
        model: Model name for task planning
    
    Returns:
        List of task dictionaries with commands, targets, and destinations
    """
    print(f"[plan_tasks] ==> Function called")
    print(f"[plan_tasks] Request: '{user_request}'")
    print(f"[plan_tasks] Model: {model}")
    print(f"[plan_tasks] Memory objects: {len(memory.list_objects())}")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build memory context
    print("[plan_tasks] Building memory context...")
    memory_context = _build_memory_context(memory)
    print(f"[plan_tasks] Memory context: {len(memory_context)} chars")
    
    # Format prompt
    prompt = TASK_SCHEDULER_PROMPT.format(
        memory_context=memory_context,
        user_request=user_request
    )
    
    print("[plan_tasks] Calling OpenAI API...")
    
    try:
        # Call GPT-5 with structured output preference
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]
                }
            ],
            text={"verbosity": "low"},
        )
        
        print("[plan_tasks] API call successful, extracting output...")
        
        # Extract output text
        output = getattr(response, "output_text", None)
        if not output:
            try:
                output = response.output[1].content[0].text
            except Exception:
                output = str(response)
        
        print(f"[plan_tasks] Got output ({len(output)} chars), parsing...")
        
        # Parse into task list
        tasks = _parse_task_list(output)
        print(f"[plan_tasks] Parsed {len(tasks)} tasks")
        
        print(f"[TaskScheduler] Generated {len(tasks)} tasks")
        for i, task in enumerate(tasks):
            print(f"  {i+1}. {task.get('command', 'UNKNOWN')}: target={task.get('target', 'N/A')}, dest={task.get('destination', 'N/A')}")
        
        return tasks
    
    except Exception as e:
        print(f"[TaskScheduler] Error during planning: {e}")
        return []


class TaskScheduler:
    """
    Convenience wrapper for task planning with state.
    Can be used as a drop-in replacement for GPTClient.
    """
    
    def __init__(self, model: str = TASK_SCHEDULER_MODEL):
        self.model = model
    
    def plan(self, user_request: str, memory: Optional[ObjectMemory] = None) -> List[Dict[str, Any]]:
        """
        Generate a task plan.
        
        Args:
            user_request: Natural language request
            memory: ObjectMemory instance (required)
        
        Returns:
            List of task dictionaries
        """
        print(f"[TaskScheduler.plan] Called with request: '{user_request}'")
        
        if memory is None:
            print("[TaskScheduler.plan] ERROR: No memory provided!")
            raise ValueError("TaskScheduler.plan() requires a memory instance")
        
        print(f"[TaskScheduler.plan] Memory has {len(memory.list_objects())} objects")
        print(f"[TaskScheduler.plan] Calling plan_tasks() with model={self.model}")
        
        return plan_tasks(user_request, memory, self.model)


__all__ = [
    "plan_tasks",
    "TaskScheduler",
    "COMMANDS",
]

