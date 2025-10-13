# vizzy/laptop/terminal_ui.py
"""
Dual terminal UI using rich library for split-screen display.
Provides separate debug and user interface sections.
"""

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
import threading
import time


class TerminalUI:
    """
    Manages a split-screen terminal interface with debug and user sections.
    
    The debug section (top 70%) shows system logs and internal state.
    The user section (bottom 30%) shows user-facing messages and prompts.
    """
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.debug_lines = []
        self.user_lines = []
        self.lock = threading.Lock()
        self._live = None
        self._update_thread = None
        self._stop_flag = False
        
    def debug(self, msg: str):
        """Add message to debug terminal (top section)."""
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            self.debug_lines.append(f"[{timestamp}] {msg}")
            self.debug_lines = self.debug_lines[-50:]  # Keep last 50 lines
    
    def user(self, msg: str):
        """Add message to user terminal (bottom section)."""
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            self.user_lines.append(f"[{timestamp}] {msg}")
            self.user_lines = self.user_lines[-20:]  # Keep last 20 lines
    
    def render(self):
        """Generate layout for display."""
        # Split screen: top 70% debug, bottom 30% user
        self.layout.split_column(
            Layout(name="debug", ratio=7),
            Layout(name="user", ratio=3)
        )
        
        with self.lock:
            debug_text = "\n".join(self.debug_lines[-30:])
            user_text = "\n".join(self.user_lines[-10:])
        
        self.layout["debug"].update(
            Panel(debug_text, title="[Debug Log]", border_style="blue")
        )
        self.layout["user"].update(
            Panel(user_text, title="[User Interface]", border_style="green")
        )
        
        return self.layout
    
    def start(self):
        """Start the live display in a separate thread."""
        if self._live is not None:
            return  # Already started
        
        def update_loop():
            with Live(self.render(), console=self.console, refresh_per_second=4) as live:
                self._live = live
                while not self._stop_flag:
                    live.update(self.render())
                    time.sleep(0.25)
        
        self._update_thread = threading.Thread(target=update_loop, daemon=True)
        self._update_thread.start()
        time.sleep(0.5)  # Give it time to initialize
    
    def stop(self):
        """Stop the live display."""
        self._stop_flag = True
        if self._update_thread:
            self._update_thread.join(timeout=2.0)
        self._live = None

