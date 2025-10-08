# vizzy/laptop/display_bus.py
# -----------------------------------------------------------------------------
# FrameBus: thread-safe, bounded queue for GUI frames.
# Workers publish frames; the main thread drains and displays them.
# This keeps all cv2.imshow()/Qt GUI calls on the main thread.
# -----------------------------------------------------------------------------

from __future__ import annotations

from queue import Queue, Full, Empty
from typing import Iterable, Optional


class FrameBus:
    def __init__(self, maxsize: int = 4):
        """
        Parameters
        ----------
        maxsize : int
            Maximum number of frames to buffer. Small to limit latency.
        """
        self._q: Queue = Queue(maxsize=maxsize)

    def publish(self, frame) -> None:
        """
        Non-blocking publish: drops the oldest frame when full
        to keep latency low (latest wins).
        """
        try:
            self._q.put_nowait(frame)
        except Full:
            try:
                # Drop one and try again
                self._q.get_nowait()
            except Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except Full:
                # If still full, just drop this frame
                pass

    def drain(self) -> Iterable:
        """
        Non-blocking drain: yields all currently queued frames (in order).
        Intended for the main/UI thread each loop tick.
        """
        out = []
        try:
            while True:
                out.append(self._q.get_nowait())
        except Empty:
            pass
        return out

    def clear(self) -> None:
        """Remove any queued frames."""
        try:
            while True:
                self._q.get_nowait()
        except Empty:
            pass
