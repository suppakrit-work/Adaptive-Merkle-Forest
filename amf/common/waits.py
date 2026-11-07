"""How long the owner's thread waited on others -- for the figures of Section V only.

A call over the network, or a wait for a pool of such calls sent out
together, is time the owner spends on no work of its own.  Whoever makes the
owner's thread wait notes the time here; a measurement subtracts it from the
wall time and what is left is the owner's own work, hashing, signing and the
wire format included.  Only waits on the main thread count, so a call made
from a pool thread is not counted twice with the wait for the pool.
"""
import threading
import time

waited = 0.0
_lock = threading.Lock()


def note(seconds: float) -> None:
    """Add a wait of the main thread."""
    global waited
    if threading.current_thread() is threading.main_thread():
        with _lock:
            waited += seconds


class waiting:
    """with waits.waiting(): ... -- the block is a wait on others."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        note(time.perf_counter() - self.t0)
        return False
