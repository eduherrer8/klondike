# Plan: Faster per-world solver via PyPy, Zobrist hashing, and a native hot-path extension

## Context

Klondike solitaire assistant (Python/Flask + vanilla JS). Monte Carlo
(`solver/montecarlo.py`) samples consistent hidden-card arrangements ("worlds")
and runs the exact solver (`solver/solver.py::solve` / `is_winnable`) on each to
estimate win probability and rank moves. Work is parallelized across worlds with
a process pool (`solver/parallel.py`); logs go to `logs/mc.log` via
`solver/mclog.py`.

**Parallelism is NOT the bottleneck and must not be the focus here.** Measured on
this machine (AMD Ryzen 9 7900X, 12 physical cores / 24 logical):

- Throughput benchmark on a heavy ~21-hidden position, fixed 8 s budget:
  `pool=12` evaluated a median of ~20 worlds; `pool=24` ~21 worlds (+5%, within
  noise). Production `logs/mc.log` corroborates: `pool=12` runs land
  `evaluated=10..23`, `pool=24` runs `evaluated=15..42` — same wide variance, no
  reliable gain. The 12-core cap in `MAX_WORKERS` is correct; do not raise it.
- `cProfile` of a single `is_winnable()` call over 36 determinized heavy worlds
  (`max_states=60_000`), by internal time (`tottime`):

  | Function | share | note |
  |---|---|---|
  | `engine.legal_moves` | ~30% | hottest single function |
  | `game.GameState.key` + its 2 list comps | ~25% | rebuilds full nested key per state |
  | `solver.is_winnable` loop body | ~10% | heap/dict/set ops |
  | `engine.apply_move` | ~6% | copy-on-write transition |
  | `_heuristic` (+ genexprs, `sum`) | ~8% | recomputed per state |
  | `cards.can_stack_tableau` | ~5% | called ~51.8M times |
  | misc (`len`, `code`, `foundation_top`, `is_won`, `can_stack_foundation`) | ~9% | |

  Total: ~211M Python function calls for 36 solves. The cost is interpreter
  overhead in tight, branchy loops over small objects — exactly the profile that
  a JIT or a native extension attacks, and that more cores cannot.

The prior doc `docs/plan-solver-speed.md` already drove a round of pure-Python
optimizations that have LANDED: int `kid` card ids, a boolean `is_winnable`
fast path, parent-pointer path reconstruction, and copy-on-write `apply_move`.
Do not redo those. This plan is the next layer: run the same code faster
(PyPy), stop rebuilding the key (Zobrist), and push the hottest 60% into native
code (Rust/Cython).

## Goal

Cut per-world solve time substantially so Monte Carlo completes many more worlds
in the same 8 s budget (reducing win-probability noise and run-to-run move
flapping). Rough targets, to be confirmed by the agent's own before/after
measurements:

- Task 1 (PyPy): 5-15x on the solve loop, zero/near-zero code change. Fastest
  signal on whether a JIT alone is "enough".
- Task 2 (Zobrist): remove most of the ~25% spent in `key()`; language-agnostic
  (benefits CPython AND PyPy AND a future native port).
- Task 3 (native hot path): 10-30x on the ported functions; biggest absolute
  win, largest effort.

These are three INDEPENDENT tracks. They can be evaluated in order and any one
can be adopted or dropped on its own. Do NOT let a later task block reporting on
an earlier one.

## Hard invariants (do NOT break — same as the solver's existing contract)

1. `solve(state, max_states) -> list[Move] | None` and
   `is_winnable(state, max_states) -> bool` keep their signatures and contracts.
   `is_winnable` must return the SAME verdict as `solve(...) is not None` for a
   given state (the suite's `check_is_winnable_matches_solve` enforces this).
2. Game rules unchanged: no splitting a face-up run; only a King to an empty
   column; draw-3 with the known cycling stock; the solver explores exactly
   `engine.legal_moves`.
3. Opaque-unknown handling in `engine`/`game`/`cards` is preserved (a face-up
   UNKNOWN blocks its column; face-down cards in a determinized world are
   concrete). Keep the guards even though determinized worlds have no UNKNOWN.
4. `GameState`, `Card`, `Move`, and `UNKNOWN` must stay **picklable** — the
   process pool ships them to workers (spawn-safe). Any hashing/caching state
   must not break pickling or leak across worlds/processes. Prefer per-`solve`
   local structures; do NOT attach mutable caches to `GameState` in a way that
   crosses process or world boundaries.
5. `solve`/`is_winnable` stay **deterministic** for a given state (no RNG, no
   dependence on dict/set iteration order or object identity). The suite's
   `check_solve_deterministic` and `check_solver_verdicts_stable` enforce this.
6. All existing `check.py` checks keep passing, plus any new ones added.

## Task 1 — PyPy (JIT), lowest effort, run first

**Idea.** Run the existing, unchanged solver under PyPy3 instead of CPython. The
hot loop (tight, call-heavy, pure Python, no C-extension dependency in the
solver itself) is PyPy's sweet spot.

**Steps.**
1. Install PyPy3 compatible with the code's syntax (3.10 features are used:
   `X | None` annotations, `match`-free). Prefer a local, non-invasive install
   (e.g. a separate `pypy` venv under `.venv-pypy/`); do NOT disturb the
   existing CPython `.venv`.
2. Confirm the solver package imports and runs under PyPy with no edits. The
   solver has no third-party deps; only `app.py` imports Flask. PyPy only needs
   to run the SOLVER + Monte Carlo + `check.py`, not necessarily Flask — but
   check whether Flask installs cleanly under PyPy too (it usually does) in case
   we later want the whole server on PyPy.
3. Run `check.py` under PyPy and confirm "All sanity checks passed." (If a check
   is CPython-specific, note it; none are expected to be.)
4. Measure: run the per-solve benchmark (snippet below) and the throughput
   benchmark under CPython vs PyPy and report the speedup. For PyPy, include a
   WARMUP run before timing (the JIT needs to warm up) and time the steady state.

**Watch-outs / report.**
- PyPy process startup and JIT warmup are real; for the MC use case (short-lived
  pooled workers that spawn, do a few solves, and may be reused) warmup cost
  matters. Measure whether the reused `ProcessPoolExecutor` workers stay alive
  long enough to amortize warmup, or whether per-call overhead eats the gain.
- `multiprocessing` under PyPy: confirm the process pool still works and
  `GameState` pickling round-trips (the suite's `check_pickle_round_trip`).
- Report: does PyPy alone get us "enough"? Give the numbers and a recommendation
  on whether to pursue Tasks 2/3 regardless.

## Task 2 — Incremental Zobrist hashing for the state key, language-agnostic

**Idea.** `GameState.key()` rebuilds a full nested tuple of ints for EVERY
generated state (~3.36M calls, ~25% of solve time, 23.5M inner iterations). Most
of a child state is identical to its parent; only the touched columns / piles
change. Replace "rebuild the whole key" with an incrementally-maintained
[Zobrist hash](https://en.wikipedia.org/wiki/Zobrist_hashing): a single integer
updated by XOR-ing in/out only the cards that moved.

**Design constraints (critical — read before coding).**
- The search uses `key()` as a dict/set key for the transposition table
  (`best_seen`), the closed set, and the parent map (`prev`). A Zobrist hash is
  a fixed-width int that can collide. Collisions in a correctness-critical
  dedupe are a real risk. Decide and document the collision policy:
  - Option A (safe default): keep the EXACT key as the dict key (correctness
    unchanged), and only use Zobrist as a fast pre-filter — but that doesn't
    remove the rebuild cost, so it is likely NOT worth it. Prefer B.
  - Option B (recommended): make the Zobrist int the dict/set key, sized at 64
    bits from a fixed, SEEDED table so it is deterministic across runs and
    across pickled worlds/processes (invariant 5). Argue the collision
    probability is negligible for the per-solve state counts here
    (`max_states <= 60_000`; even a few hundred thousand states keeps 64-bit
    collision odds astronomically low). Document this explicitly as the accepted
    risk, with the math, in the plan's report.
- The Zobrist table (random constants per (slot, card-identity, orientation),
  plus foundation-count and stock/waste contributions) must be a MODULE-LEVEL
  CONSTANT seeded deterministically, identical in every process (so pickled
  worlds agree). Do NOT regenerate per `GameState` or per process with a fresh
  RNG.
- The hash must be a pure function of position (same encoding the current
  `key()` captures: per-column card identities + orientation, foundation pile
  LENGTHS, stock order, waste order), so equal positions hash equal and the
  determinism/dedupe checks still hold.
- Incremental update lives in `apply_move` (the only place a state is derived):
  given the parent's hash, XOR out what left a slot and XOR in what arrived.
  Keep `apply_move`'s copy-on-write and picklability. Store the running hash as
  a plain int field on `GameState` (picklable; include it in `clone`).
- Keep a `key()` (or equivalent) that still produces the EXACT canonical tuple
  for any code path or test that needs a true identity (e.g. the collision-free
  equality spot-check), OR prove every consumer is fine with the hash. The
  existing `check_cheap_key_dedupes` must still pass or be updated to the new
  contract with justification.

**Steps.**
1. Add the seeded Zobrist constant table (module-level, deterministic).
2. Compute the full hash once for a freshly built/determinized state.
3. Maintain it incrementally in each branch of `apply_move`; carry it through
   `clone`; include it in pickling.
4. Switch the solver's `best_seen` / closed set / `prev` keys to the hash
   (Option B), or justify an alternative.
5. Add/adjust checks: equal states hash equal; a set of distinct states hash
   distinct (spot check); determinism across a pickle round-trip; verdicts
   unchanged on the fixed verdict-stability worlds.

**Watch-outs.** This is the task most able to silently change RESULTS if a
collision or a non-deterministic table slips in. The agent MUST verify the
winnable worlds in the benchmark stay winnable AND that returned winning lines
still reach `is_won()` when applied (the suite already has
`check_winning_lines_reach_won` and `check_solver_verdicts_stable`).

## Task 3 — Native extension for the hottest ~60% (Rust via PyO3, or Cython)

**Idea.** `legal_moves` (~30%) + `key`/hash (~25%) + `apply_move` (~6%) are ~60%
of solve time in three functions. Port just those (and the trivial
`can_stack_*`, which they call tens of millions of times) to a native extension,
leaving the Flask app, resolver, strategy, storage, and MC orchestration in
Python. This is the surgical, highest-absolute-gain option and the largest
effort.

**Decision to make and document: Rust (PyO3/maturin) vs Cython.**
- Rust/PyO3: best raw speed and memory control; adds a Rust toolchain +
  `maturin` build step; a compiled wheel per platform.
- Cython: smaller leap from the existing Python (annotate types, `cdef`);
  easier incremental porting; needs a C compiler and a build step.
Pick one, justify it, and keep a PURE-PYTHON FALLBACK so the project still runs
if the extension isn't built (import the native module, fall back to the current
Python implementations on ImportError). This preserves the invariant that
`check.py` and `app.py` run on a clean checkout.

**Scope boundaries (do NOT over-reach).**
- Port the per-node primitives, NOT the whole search. The best-first loop
  (`heapq`, dicts, path reconstruction) can stay in Python at first; measure
  whether moving it native later is worth it. Keeping the loop in Python keeps
  the diff reviewable and preserves determinism guarantees.
- The native move generator must produce moves in the SAME ORDER as the current
  `legal_moves` (search order affects which winning line is found; invariants
  5-6). Mirror the exact ordering in `engine.legal_moves` (foundation moves,
  waste-to-foundation, tableau runs by source then dest, waste-to-tableau,
  draw).
- Data crossing the boundary must stay cheap: pass compact ints (the existing
  `kid` encoding), not Python `Card` objects, into the native layer.

**Steps.**
1. Stand up the build (maturin/Cython) behind a feature-detected import.
2. Port `can_stack_tableau`/`can_stack_foundation`, then `legal_moves`, then the
   key/hash, then `apply_move` — one at a time, re-running `check.py` after each
   so a regression is localized.
3. Benchmark after each port; stop if the gains plateau below the added
   complexity.

**Watch-outs.** Build/packaging complexity, platform wheels, and the
pickling/parallel boundary (native objects must not need to cross the process
pool — keep `GameState` a plain Python object; the native functions operate on
it or on extracted int arrays). Determinism and move-order parity are the
correctness traps.

## Validation / benchmarks (run for EACH task; use `.venv/bin/python`, and PyPy where relevant)

1. **Correctness first.** `.venv/bin/python check.py` must end
   "All sanity checks passed." For Task 1 also run `check.py` under PyPy. Add new
   checks where a task introduces new behavior (Zobrist equality/collision spot
   check; native/py fallback parity — the native and Python `legal_moves` must
   return identical move lists on a fixed set of states).
2. **Per-solve benchmark (before vs after).** Deterministically build the 12
   worlds of a fixed seed, solve each at `max_states=60_000`, report per-world
   time, total, and the WIN/no-sol outcome per world (outcomes must be
   unchanged). Snippet below.
3. **Throughput (end-to-end).** Start the server (`.venv/bin/python app.py`,
   dev server on :5001), run a heavy ~21-hidden analysis at the default budget,
   and read `logs/mc.log`: the `estimate DONE evaluated=X/...` line reports how
   many worlds completed. Compare X before vs after. Keep `pool` fixed (12)
   across the comparison so the variable under test is the solver, not the pool.
4. **Determinism + pickling** re-verified after Tasks 2 and 3
   (`check_solve_deterministic`, `check_pickle_round_trip`).

### Benchmark snippet (deterministic heavy worlds)

```python
import time, random
from solver.cards import full_deck, UNKNOWN
from solver.game import GameState
from solver.montecarlo import _hidden_slots, _unknown_pool, _determinize
from solver.solver import solve, is_winnable

def heavy(seed):
    rng=random.Random(seed); deck=full_deck(); rng.shuffle(deck)
    gs=GameState(draw_count=3); idx=0
    for n in range(1,8):
        gs.tableau[n-1]=[(deck[idx+i], i==n-1) for i in range(n)]; idx+=n
    gs.stock=deck[idx:][::-1]
    for col in gs.tableau:
        for pi,(c,up) in enumerate(col):
            if not up: col[pi]=(UNKNOWN,False)
    return gs

st=heavy(2); slots=_hidden_slots(st); pool=_unknown_pool(st)
tot=0.0
for i in range(12):
    w=_determinize(st, slots, pool, random.Random(i))
    t=time.perf_counter(); sol=solve(w, max_states=60000); dt=time.perf_counter()-t
    tot+=dt
    print(i, f"{dt:.3f}s", "WIN" if sol else "no-sol")
print(f"total {tot:.3f}s")
```

## Deliverables (report back to the orchestrating agent, PER TASK)

For each of Tasks 1-3 attempted:
- Per-file summary of changes (and any new build steps / install instructions).
- The per-solve benchmark table BEFORE vs AFTER (same seed), with identical
  WIN/no-sol outcomes confirmed.
- Throughput: worlds completed in the default heavy-position budget at `pool=12`,
  before vs after (from `logs/mc.log`).
- `check.py` tail showing all pass (CPython; and under PyPy for Task 1),
  including any new checks.
- Confirmation that `GameState` still pickles and `solve`/`is_winnable` stay
  deterministic and return unchanged verdicts.
- For Task 2: the explicit collision-risk argument and the chosen key policy.
- For Task 3: the Rust-vs-Cython decision with justification, and confirmation
  the pure-Python fallback still runs on a clean checkout.
- Any invariant that could not be preserved, and why.
- A clear recommendation: which of the three to adopt, in what order, and
  whether later tasks are still worth it given earlier results.

## Rules of engagement

- Treat the three tasks as independent; report each on its own merits. If Task 1
  (PyPy) already gets us where we need, say so explicitly and flag Tasks 2/3 as
  optional.
- Do NOT raise `MAX_WORKERS` or otherwise chase parallelism — it has been
  measured as a dead end on this machine.
- Do NOT change search results beyond acceptable search-order effects. Winnable
  stays winnable; returned winning lines must reach `is_won()` when applied.
- Do NOT commit. Leave changes in the working tree for review. If a task needs a
  new directory (e.g. `.venv-pypy/`, a Rust crate), keep it self-contained and
  add it to `.gitignore` as appropriate.
- Keep any throwaway benchmark/profile scripts out of the committed tree (or
  clearly mark and remove them) — the previous round left none behind; match
  that.

## How to run

```
.venv/bin/python check.py      # sanity suite (CPython)
.venv/bin/python app.py        # dev server on :5001
# PyPy (Task 1), once installed, e.g.:
.venv-pypy/bin/pypy check.py
```
Clear test rows after manual testing: `DELETE FROM games;` in `games.db`.
Logs: `logs/mc.log` (grep `estimate DONE` for `evaluated=X/...`).
```
