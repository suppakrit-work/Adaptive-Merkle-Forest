"""How long the owner waited on others -- for the figures of Section V only.

A call over the network is time the owner spends on no work of its own, and
so is the time its main thread spends waiting for a pool of such calls sent
out together, except for the work the pool threads did meanwhile.  Whoever
makes the owner wait notes it here; a measurement subtracts `waited` from the
wall time and what is left is the owner's own work: hashing, signing, the
wire format included, on whichever thread it ran.

    waits.note(dt)         a network call of dt seconds on this thread
    with waits.waiting():  the main thread waits for a pool
    waits.pooled(fn)       a pool task: its own work (wall minus its own
                           network calls) is subtracted from the pool wait
"""
import threading
import time

waited = 0.0                      # seconds the main thread spent waiting, net of work done meanwhile
_pool_work = 0.0                  # work done inside pool tasks (wall minus their network calls)
_lock = threading.Lock()
_local = threading.local()


def note(seconds: float) -> None:
    """A network call on the current thread."""
    global waited
    if threading.current_thread() is threading.main_thread():
        with _lock:
            waited += seconds
    else:
        _local.net = getattr(_local, "net", 0.0) + seconds


def pooled(fn):
    """Wrap a task given to a pool so that the work it does is not mistaken for waiting."""
    def run(*args, **kwargs):
        global _pool_work
        before, _local.net = getattr(_local, "net", 0.0), 0.0
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            work = max(time.perf_counter() - t0 - _local.net, 0.0)
            _local.net = before
            with _lock:
                _pool_work += work
    return run


class waiting:
    """with waits.waiting(): ... -- the main thread waits for pooled tasks."""

    def __enter__(self):
        with _lock:
            self.work0 = _pool_work
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        global waited
        wall = time.perf_counter() - self.t0
        if threading.current_thread() is threading.main_thread():
            with _lock:
                waited += max(wall - (_pool_work - self.work0), 0.0)
        return False
