# Plan: Speed up the per-world exact solver

## Context

Klondike solitaire ASSISTANT (Python/Flask + vanilla JS). Monte Carlo
(`solver/montecarlo.py`) samples consistent hidden-card arrangements ("worlds")
and runs the exact solver (`solver/solver.py::solve`) on each to estimate win
probability and rank moves. It is now **parallelized** across worlds with a
process pool (`solver/parallel.py`) and logs to `logs/mc.log` via
`solver/mclog.py`.

**The remaining bottleneck is the per-world solver, not parallelism.** Measured
on a heavy ~21-hidden position (12 sampled worlds, `max_states=60_000`):

```
winnable worlds:        ~0.01 - 0.31 s each
unwinnable/hard worlds: ~1.5 - 1.7 s each  (run the FULL state budget, then give up)
12 worlds = 3.8 s total; ~3.3 s of it was just 2 budget-exhausting worlds
```

So the few worlds that exhaust `max_states` dominate wall-clock. In a 5 s MC
budget across 12 cores, that means only ~7-11 of 120 target samples complete on
heavy positions — which makes the win-probability noisy (±0.2-0.3) and the move
recommendation vary run to run (the user's "re-run and it solves" symptom).

cProfile of a single `solve` (per-node hotspots, in order):
`engine.legal_moves` (~39%), `game.GameState.key` and its generator expressions,
`game.column_face_up_run`, `cards.is_unknown` (called ~24k times), plus
`apply_move` cloning the whole state and `_heuristic` recomputed per node.

## Goal

Make each `solve()` call substantially faster — especially the long,
budget-exhausting searches — so Monte Carlo completes many more worlds in the
same wall-clock. Target: at least a **2-3x** per-solve speedup on heavy worlds,
which should roughly 2-3x the samples completed in the MC time budget.

Do NOT change the solver's RESULTS (which deals it reports winnable, the moves
it returns) beyond acceptable search-order effects. The win/lose verdict for a
fully-known world must stay the same where it was previously conclusive.

## Hard invariants (do NOT break)

1. `solve(state, max_states) -> list[Move] | None` keeps its signature and
   contract: returns a winning move list, or None if not found within budget.
   `is_won()` state returns `[]`.
2. The game rules are unchanged: no splitting a face-up run; only a King to an
   empty column; draw-3 with the known cycling stock; conservative nothing —
   the solver explores all legal moves from `engine.legal_moves`.
3. Opaque unknowns: a face-up UNKNOWN card blocks its column and can't move
   (see `engine.legal_moves` / `game.column_face_up_run` / `cards.is_unknown`).
   Face-DOWN cards in a fully-determinized world are concrete (identities
   assigned); `solve` operates on concrete worlds, but the engine must still
   correctly handle any UNKNOWN that appears (it shouldn't in a determinized
   world, but keep the guards).
4. `GameState` must stay **picklable** (multiprocessing ships it to workers).
   Any caching/transposition structure must NOT be stored on the GameState in a
   way that breaks pickling or leaks across worlds/processes. Prefer per-`solve`
   local structures.
5. All existing `check.py` sanity checks must still pass, plus new ones you add.
6. Determinism: `solve` is currently deterministic for a given state (no RNG).
   Keep it deterministic.

## Suggested work (evidence-based; the agent may adjust with justification)

### A. Cheaper per-node work (speeds up every solve)

1. **`GameState.key()`** is rebuilt from nested generator expressions on every
   generated state (757 calls building 6k+ tuples in the profile). Options:
   - Build the key more cheaply (avoid per-card `is_unknown` in the hot path;
     precompute rank/suit tuples).
   - Represent a card's identity for the key as a small int (e.g.
     `rank * 4 + suit_index`, with a sentinel for UNKNOWN) to avoid tuple+attr
     overhead.
2. **`is_unknown`** is called ~24k times. It does `isinstance(c, _Unknown)`.
   Consider a faster check (e.g. compare against the singleton `UNKNOWN` by
   identity: `c is UNKNOWN`), or cache per-card.
3. **`engine.legal_moves` / `column_face_up_run`** are the top cost. Look for
   redundant recomputation (e.g. `column_face_up_run` called multiple times per
   column per node) and compute the movable run once per column per node.
4. **`apply_move` clones the whole state** for every candidate move, including
   moves that are immediately discarded as duplicates. Consider computing the
   resulting state key WITHOUT a full clone first, and only clone when the state
   is actually going to be enqueued (not a known-worse duplicate). Be careful to
   keep correctness.
5. **Avoid carrying the full `path` in every heap entry.** The current code does
   `path + [move]` (O(path) copy) for every pushed node, and stores the whole
   path on the heap. Prefer a parent-pointer / came-from map
   (`prev[state_key] = (parent_key, move)`) and reconstruct the path only when a
   win is found. This removes a major allocation cost in long searches.

### B. Transposition table that prunes harder (helps the long searches)

- The existing `best_seen: dict[key -> best_path_len]` is a transposition table
  already, but it's keyed by the expensive `key()` and still enqueues many
  states. Improve it:
  - Use the cheaper key from A.
  - When a state key is seen at an equal-or-worse depth, skip BEFORE cloning /
    before pushing (prune earlier).
  - Consider a visited/closed set so popped states aren't re-expanded.

### C. Optional: a boolean "winnable?" fast path

- `position_estimate` only needs yes/no winnable, not the move list. A dedicated
  `is_winnable(state, max_states)` that skips path bookkeeping entirely could be
  cheaper for that path. OPTIONAL — only if A/B don't already close the gap, and
  only if it doesn't duplicate too much logic. If added, `montecarlo`'s
  `eval_world_winnable` worker (in `solver/parallel.py`) could call it.

## Validation / benchmarks to run and report

Use the venv Python: `.venv/bin/python`.

1. **Correctness first.** `.venv/bin/python check.py` must end
   "All sanity checks passed." Add new checks:
   - `solve` returns the SAME win/lose verdict as before on a set of fixed
     fully-known worlds (capture a few winnable and a few budget-exhausting
     worlds; assert winnable stays winnable). Build them deterministically with
     a fixed seed (see the bench snippet below).
   - `GameState` still pickles (round-trip) after any key/representation change.
   - The cheaper `key()` still DEDUPES correctly: two genuinely-equal states
     produce equal keys; two different states produce different keys (spot
     check).
2. **Per-solve benchmark.** Reproduce the measurement that motivated this:
   deterministically build the 12 worlds of seed 2 (snippet below), solve each
   at `max_states=60_000`, report per-world time and total, BEFORE vs AFTER.
   Target: total down from ~3.8 s to ideally <=1.5 s (>=2x), with the SAME
   winnable/no-sol outcomes per world.
3. **End-to-end sample count.** Start the server (`.venv/bin/python app.py`),
   run a heavy position (21 hidden) analysis for the default 5 s budget, and
   report how many worlds completed (read `logs/mc.log`: the
   `estimate DONE evaluated=X/120` line). Compare X before vs after; expect a
   meaningful increase.

### Benchmark snippet (deterministic heavy worlds)

```python
import time, random
from solver.cards import full_deck, UNKNOWN
from solver.game import GameState
from solver.montecarlo import _hidden_slots, _unknown_pool, _determinize
from solver.solver import solve

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
for i in range(12):
    w=_determinize(st, slots, pool, random.Random(i))
    t=time.time(); sol=solve(w, max_states=60000); dt=time.time()-t
    print(i, f"{dt:.2f}s", "WIN" if sol else "no-sol")
```

## Risks / watch-outs

- **Don't change search results.** If you reorder exploration or change the
  transposition policy, verify the winnable worlds in the benchmark stay
  winnable and the returned move lists still reach a won state (apply them with
  `engine.apply_move` and assert `is_won()`).
- **Pickling**: if you change the card/state representation for the key, re-run
  the pickle round-trip test; the process pool depends on it.
- **No shared mutable global caches** across worlds/processes (would leak state
  or break pickling). Keep transposition tables local to each `solve` call.
- **`solve` is used in two very different ways**: by MC (many short-budget calls
  in workers) and by `exact_win_plan`/`is_dead_stock_cycle` (correctness-
  critical). Speedups must not alter the latter's conclusions.

## Deliverables (report back to the orchestrating agent)

- Per-file summary of changes.
- The per-solve benchmark table BEFORE vs AFTER (12 worlds, seed 2), with the
  same WIN/no-sol outcomes confirmed.
- End-to-end: worlds completed in the default 5 s heavy-position budget, before
  vs after (from `logs/mc.log`).
- `check.py` tail showing all pass, including the new checks.
- Confirmation `GameState` still pickles and `solve` stays deterministic.
- Any invariant you could not preserve and why.
- Do NOT commit; leave changes in the working tree for review.

## How to run

```
.venv/bin/python check.py      # sanity suite
.venv/bin/python app.py        # dev server on :5001
```
Clear test rows after manual testing: `DELETE FROM games;` in `games.db`.
Logs: `logs/mc.log`.
