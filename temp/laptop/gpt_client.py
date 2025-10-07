# vizzy/laptop/gpt_client.py
# -----------------------------------------------------------------------------
# Stub GPT client & plan schema.
# Replace GPTClient.plan() with your real API call when ready.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal, List, Dict, Any, Optional


Priority = Literal["low", "normal", "high", "urgent"]
CmdKind = Literal["look_for", "center_on", "move_to_pose", "track", "pick", "place"]


@dataclass(frozen=True)
class HighLevelCmd:
    kind: CmdKind
    params: Dict[str, Any]


@dataclass(frozen=True)
class Plan:
    priority: Priority
    requires_scan: bool
    interrupt_ok: bool
    actions: List[HighLevelCmd]


class GPTClient:
    """
    Minimal façade. In production, call your hosted GPT endpoint here.
    Environment hooks (optional now):
      - VIZZY_GPT_ENDPOINT
      - VIZZY_GPT_KEY
    """

    def __init__(self, endpoint: Optional[str] = None, api_key: Optional[str] = None):
        self.endpoint = endpoint or os.getenv("VIZZY_GPT_ENDPOINT", "")
        self.api_key = api_key or os.getenv("VIZZY_GPT_KEY", "")

    # ------------------------- simple local parser (stub) ---------------------

    def plan(self, text: str) -> Plan:
        """
        Return a structured Plan. This is a *stub* that does no network I/O.
        It recognizes a few common intents to keep the pipeline testable.
        """
        t = text.strip().lower()

        # center on <class>
        m = re.search(r"(center|focus)\s+on\s+(the\s+)?(?P<cls>[a-z0-9_ \-]+)", t)
        if m:
            cls = m.group("cls").strip()
            return Plan(
                priority="normal",
                requires_scan=True,
                interrupt_ok=False,
                actions=[HighLevelCmd(kind="center_on", params={"class_name": cls})],
            )

        # move to pose: "move to pwm 1450 1200" or "goto 1500, 1300"
        m = re.search(r"(move\s+to|goto)\s+(pwm\s+)?(?P<btm>\d{3,4})[,\s]+(?P<top>\d{3,4})", t)
        if m:
            return Plan(
                priority="normal",
                requires_scan=False,
                interrupt_ok=False,
                actions=[
                    HighLevelCmd(
                        kind="move_to_pose",
                        params={"pwm_btm": int(m.group("btm")), "pwm_top": int(m.group("top")), "slew_ms": 600},
                    )
                ],
            )

        # look for / find <class>
        m = re.search(r"(look\s+for|find)\s+(?P<cls>[a-z0-9_ \-]+)", t)
        if m:
            cls = m.group("cls").strip()
            return Plan(
                priority="normal",
                requires_scan=True,
                interrupt_ok=False,
                actions=[HighLevelCmd(kind="look_for", params={"class_name": cls})],
            )

        # default fallback: do nothing (but remain valid)
        return Plan(
            priority="low",
            requires_scan=False,
            interrupt_ok=False,
            actions=[],
        )
