# vizzy/laptop/llm_semantics.py
# -----------------------------------------------------------------------------
# LLM semantic enrichment for centered objects using GPT-5 vision API.
# -----------------------------------------------------------------------------

import os
import json
import re
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

IMAGE_PROCESS_MODEL = os.getenv("IMAGE_PROCESS_MODEL", "gpt-5")

SEMANTICS_PROMPT = (
    "Identify only the center-most object. Return strictly JSON with keys: "
    '{"name","material","color","unique_attributes","grasp_position","grasp_xy"}. '
    '"grasp_xy" must be a two-element array of pixel coordinates [x, y]. '
    "No extra text—JSON only."
)


def upload_image(image_path: str, *, api_key: Optional[str] = None) -> str:
    """
    Upload an image file to OpenAI 'vision' storage; return file_id.
    
    Args:
        image_path: Path to the image file
        api_key: Optional OpenAI API key (defaults to env var)
    
    Returns:
        OpenAI file ID for the uploaded image
    """
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    with open(image_path, "rb") as f:
        res = client.files.create(file=f, purpose="vision")
    return res.id


def _try_parse_semantics(text: str) -> Dict[str, Any]:
    """
    Robustly parse semantics from model output.
    Accepts raw JSON, or JSON inside code fences. Falls back to minimal mapping.
    
    Args:
        text: Raw text output from the model
    
    Returns:
        Dictionary with semantic fields (name, material, color, etc.)
    """
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


def call_llm_for_semantics(file_id: str, model: str = IMAGE_PROCESS_MODEL) -> Dict[str, Any]:
    """
    Call Responses API asking for strict JSON semantics.
    Uses response_format when available; falls back gracefully.
    
    Args:
        file_id: OpenAI file ID of the uploaded image
        model: Model name (defaults to IMAGE_PROCESS_MODEL)
    
    Returns:
        Dictionary of semantic attributes
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

