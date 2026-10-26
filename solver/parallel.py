"""Process-pool plumbing for parallel Monte Carlo world evaluation.

Monte Carlo determinization is embarrassingly parallel: every sampled world is
solved independently and nothing is shared. This module owns a single, lazily
created, reused ``ProcessPoolExecutor`` and the top-level worker functions the
pool runs.

Design constraints (see docs/plan-parallel-mc.md):

* macOS defaults to the "spawn" start method, so each worker re-imports this
  module. It MUST import cleanly with no heavy/side-effecting work at import
  time. In particular the pool is created LAZILY, never at import.
* Worker functions must be top-level (picklable) — no closures/lambdas.
* The Flask debug reloader spawns a child process; creating the pool lazily on
  first use (and only in a real request) keeps us from spinning up a pool in
  the reloader's supervisor process.
* Graceful fallback: if pool creation fails, callers fall back to a clean
  single-process path. Nothing here may hang or crash the request.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ProcessPoolExecutor

from .engine import apply_move
from .game import GameState, Move
from .solver import is_winnable, solve

# Hard cap on worker processes. The target machine has 12 physical cores; going
# wider than the core count just adds contention for a CPU-bound pure-Python
# workload.
MAX_WORKERS = 12

# Module-level singletons, created lazily under a lock so concurrent Flask
# requests don't race to build two pools.
_pool: ProcessPoolExecutor | None = None
_pool_workers: int = 0
_lock = threading.Lock()


def pool_size() -> int:
    """Number of worker processes a pool would use on this machine."""
    cpu = os.cpu_count() or 1
    return max(1, min(cpu, MAX_WORKERS))


def get_pool() -> ProcessPoolExecutor | None:
    """Return the shared process pool, creating it on first use.

    Returns None when parallelism is unavailable or pointless (single core, or
    pool creation failed) so the caller can take the sequential path. Never
    raises — a failure degrades to None.
    """
    global _pool, _pool_workers

    workers = pool_size()
    if workers <= 1:
        return None

    if _pool is not None:
        return _pool

    with _lock:
        if _pool is not None:  # built while we waited for the lock
            return _pool
        try:
            _pool = ProcessPoolExecutor(max_workers=workers)
            _pool_workers = workers
        except Exception:
            # OS limits, sandbox, etc. Degrade to sequential.
            _pool = None
            _pool_workers = 0
            return None
    return _pool


def shutdown_pool() -> None:
    """Tear down the shared pool (used by a clean-exit hook)."""
    global _pool, _pool_workers
    with _lock:
        if _pool is not None:
            try:
                _pool.shutdown(wait=False, cancel_futures=True)
            except TypeError:  # pragma: no cover - older Python without cancel_futures
                _pool.shutdown(wait=False)
            except Exception:  # pragma: no cover - best-effort teardown
                pass
            _pool = None
            _pool_workers = 0


# ---- worker payloads --------------------------------------------------------
#
# Workers receive plain picklable data: a determinized GameState (hidden-card
# identities already assigned in the parent), the moves to test as a tuple of
# Move objects, and the per-world solve budget. They return plain tuples/bools
# so nothing exotic crosses the process boundary.


def eval_world_moves(world: GameState, moves: tuple[Move, ...],
                     solve_budget: int) -> tuple[bool, ...]:
    """Per-move worker: for one determinized world, return a bool per move —
    True if the position after that move is solvable within the budget.

    Pure and side-effect free: safe to run in any worker process.
    """
    results = []
    for mv in moves:
        after = apply_move(world, mv)
        results.append(solve(after, max_states=solve_budget) is not None)
    return tuple(results)


def eval_world_winnable(world: GameState, solve_budget: int) -> bool:
    """Position worker: is this determinized world winnable at all?

    Uses the boolean `is_winnable` fast path (no winning-line bookkeeping),
    which returns the same verdict as `solve(...) is not None` but skips the
    path reconstruction the caller would discard.
    """
    return is_winnable(world, max_states=solve_budget)
