"""
Terminal UI utilities for displaying loading animations and progress.
"""

import sys
import time
import threading
from typing import Optional


# Braille characters for spinning animation
# These create a smooth rotation effect
BRAILLE_SPINNER = ["⣾", "⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽"]


class LoadingIndicator:
    """
    Displays a loading animation with spinner and elapsed time counter.
    Uses a background thread for smooth animation.

    Usage:
        loader = LoadingIndicator("Thinking")
        loader.start()
        # ... do work ...
        loader.stop()

    Or as context manager:
        with LoadingIndicator("Thinking"):
            # ... do work ...
    """

    def __init__(self, message: str = "Thinking", update_interval: float = 0.1):
        self.message = message
        self.update_interval = update_interval
        self.spinner_index = 0
        self.start_time: Optional[float] = None
        self.running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_line_length = 0
        self._stopped = False

    def _render(self, elapsed: float) -> str:
        """Render the loading indicator line."""
        spinner = BRAILLE_SPINNER[self.spinner_index]

        # Format elapsed time as MM:SS or SS
        total_seconds = int(elapsed)
        minutes = total_seconds // 60
        seconds = total_seconds % 60

        if minutes >= 1:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{total_seconds}s"

        # Build the display string with cycling dots
        DOTS = ["   ", ".  ", ".. ", "..."]
        dots = DOTS[int(elapsed * 2) % 4]
        line = f"\r{spinner} {self.message}{dots} [{time_str}]"

        return line

    def _clear_line(self):
        """Clear the current line."""
        sys.stdout.write("\r" + " " * self._last_line_length + "\r")
        sys.stdout.flush()

    def _animate(self):
        """Animation loop running in background thread."""
        while not self._stop_event.is_set():
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                line = self._render(elapsed)
                sys.stdout.write(line)
                sys.stdout.flush()
                self._last_line_length = len(line)
                self.spinner_index = (self.spinner_index + 1) % len(BRAILLE_SPINNER)

            self._stop_event.wait(self.update_interval)

    def start(self):
        """Start the loading animation."""
        if self._stopped:
            return

        self.running = True
        self.start_time = time.time()
        self._stop_event.clear()

        # Start animation thread
        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        """Stop the loading animation and clear the line."""
        if self._stopped:
            return

        self.running = False
        self._stopped = True

        # Signal thread to stop
        self._stop_event.set()

        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

        # Clear the loading line
        self._clear_line()

        # Optionally show completion message
        if not success:
            sys.stdout.write("⚠ Error\n")
            sys.stdout.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type):
        self.stop(success=(exc_type is None))
        return False


def stream_with_indicator(generator, message: str = "Thinking"):
    """
    Stream tokens from a generator while showing a loading indicator.
    The indicator shows until the first token arrives, then clears for output.

    Usage:
        for token in stream_with_indicator(generate_streaming(...), "Generating"):
            print(token, end="", flush=True)

    Args:
        generator: The generator yielding tokens
        message: Message to display while waiting

    Yields:
        Tokens from the generator
    """
    loader = LoadingIndicator(message)
    loader.start()

    try:
        for item in generator:
            # Stop loader on first item
            if loader.running:
                loader.stop()
            yield item
    finally:
        if loader.running:
            loader.stop()
