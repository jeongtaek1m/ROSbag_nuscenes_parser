import sys
import os
import struct
import math
import argparse
import json
import time as _time

# ---------------------------------------------------------------------------
# Simple progress bar (no external dependencies)
# ---------------------------------------------------------------------------
class ProgressBar:
    """Minimal progress bar using carriage return."""

    def __init__(self, total, prefix='', width=40):
        self.total = total
        self.prefix = prefix
        self.width = width
        self.current = 0
        self.start_time = _time.time()
        self._last_print = 0

    def update(self, n=1):
        self.current += n
        now = _time.time()
        # Throttle prints to ~10 Hz to avoid I/O bottleneck
        if now - self._last_print < 0.1 and self.current < self.total:
            return
        self._last_print = now
        self._draw()

    def _draw(self):
        elapsed = _time.time() - self.start_time
        frac = self.current / self.total if self.total > 0 else 1.0
        filled = int(self.width * frac)
        bar = '#' * filled + '-' * (self.width - filled)
        pct = frac * 100

        if elapsed > 0 and self.current > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed
            time_str = ' ETA {:.0f}s'.format(eta)
        else:
            time_str = ''

        line = '\r{} [{}] {:5.1f}% ({}/{}){}  '.format(
            self.prefix, bar, pct, self.current, self.total, time_str)
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self):
        self.current = self.total
        self._draw()
        sys.stderr.write('\n')
        sys.stderr.flush()
