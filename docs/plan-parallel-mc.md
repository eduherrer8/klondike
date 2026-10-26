# Plan: Parallel Monte Carlo + Racing Elimination + Deep Analyze

## Context

This is a Klondike solitaire **assistant** (not a game). The user enters a
mid-game position with face-down cards marked `?`; the app recommends moves and
estimates win probability. The stock is fully known, so the ONLY uncertainty is
the ORDER of the face-down tableau cards. Their identities are fixed by
elimination (full deck minus everything with a known location).

Two recommenders exist today:
- `solver/strategy.py` — fast greedy planner over the known/opaque state.
- `solver/montecarlo.py` — Monte Carlo determinization: sample consistent
  hidden-card arrangements, run the exact solver (`solver/solver.py::solve`) on
  each, aggregate win rates per move and an overall win probability.

Monte Carlo is the strong method but slow. Users report that some solvable
games get a bad recommendation on the first analysis but solve when re-run.
Root cause: on heavy positions (many hidden cards) the time cap only evaluates
a few sampled worlds, so BOTH the win probability and the move RANKING are
noisy — a different random sample (a re-run) happens to pick a winning move.

## Goal

Make Monte Carlo fast enough to sample far more worlds in the same wall-clock,
so the first-run recommendation is stable (no "re-run and it works"). Achieve
this with (1) process-level parallelism across sampled worlds, (2) racing /
early elimination of clearly-losing moves, and (3) a "Deep Analyze" tier that
spends the saved time on many more samples.

The machine has **12 physical cores** (`os.cpu_count() == 12`). Monte Carlo is
embarrassingly parallel (each world solves independently, nothing shared), so
expect roughly **8-10x** real speedup.

## Hard requirements / invariants (do NOT break these)

1. **Correctness of card accounting.** Face-down card identities are fixed by
   elimination; only their order varies between worlds. See
   `montecarlo._hidden_slots`, `_unknown_pool`, `_determinize`, `_build_world`.
2. **Game-over semantics.** A position is game over when there are no legal
   moves OR it is a `strategy.is_dead_stock_cycle`. MC must keep reporting both
   (already wired into `estimate` and `position_estimate`). Preserve this.
3. **Proven-win shortcut.** `montecarlo.exact_win_plan` and the
   `app._compute_plan` ordering (exact win → MC → greedy) must be preserved.
4. **Confidence intervals.** Keep the Wilson interval (`_wilson_interval`) and
   the `winMargin`/`exact`/`timedOut` fields in responses.
5. **wins <= trials always.** A prior crash came from `wins > trials` →
   complex sqrt. On abort/partial evaluation, never leave a move's win count
   above the number of fully-evaluated worlds. (See the existing roll-back in
   `estimate`.) Parallel aggregation must uphold this.
6. **All existing sanity checks in `check.py` must still pass**, plus new ones
   you add. Run: `.venv/bin/python check.py`.
7. **Graceful single-process fallback.** If multiprocessing is unavailable or a
   worker errors, fall back to the current in-process path. Never hang or crash
   the Flask request.
8. **Determinism option.** Support an optional RNG seed so the same position +
   seed gives the same answer (helps debugging and kills "different every run").

## Current API surface (reference)

`solver/montecarlo.py`:
- `position_estimate(state, samples=60, solve_budget=60_000, rng=None, time_budget_s=8.0) -> dict`
  Cheap: one solve per world, "is this world winnable at all?". No per-move loop.
- `estimate(state, samples=60, solve_budget=60_000, rng=None, time_budget_s=8.0) -> dict`
  Per-move: for each legal move, fraction of worlds winnable after that move.
  Returns `perMove` (sorted best-first, each with `winRate`, `margin`, `wins`,
  `trials`), `winProbability`, `winMargin`, `exact`, `mode`, `samples`,
  `timedOut`, `best`, `gameOver`, and `forced` (single-move pruning).
- `mc_plan_steps(state, n=3, samples=60, solve_budget=60_000, rng=None, time_budget_s=8.0) -> dict`
  Wraps `estimate` into a step plan (shape matches `strategy.plan_steps` plus MC
  fields). Used when MC drives recommendations.
- `exact_win_plan(...)`, `_wilson_interval`, `_choose_effort`, `_world_stream`,
  `_build_world`, `_determinize`, `_hidden_slots`, `_unknown_pool`.

`app.py` integration:
- Constants: `MC_SAMPLES=60`, `MC_SOLVE_BUDGET=30_000`, `MC_TIME_BUDGET_S=8.0`.
- `_compute_plan(game, n)` → exact_win_plan, else mc_plan_steps (if `game.mc_mode`),
  else `strategy.plan_steps`.
- Endpoints: `POST /api/analyze/<id>` (per-move), `POST /api/winprob/<id>`
  (cheap position), `POST /api/mcmode/<id>` (toggle), plus the
  accept/reveal/steps flow that calls `_compute_plan`.
- The game object (`Game` dataclass) holds `state`, `layout`, `mc_mode`, `history`.

The per-world solver is `solver.solver.solve(state, max_states=...) -> list[Move] | None`.
`GameState` is a plain dataclass of lists/dicts of `Card` (also a lightweight
class) — verify it is picklable for multiprocessing (it should be; confirm with
a round-trip `pickle.dumps/loads` in a test).

## Chosen decisions (from product owner)

1. **Default MC path uses a reused process pool automatically** (no separate
   opt-in for parallelism) + **racing elimination**.
2. **Sample tiers:**
   - **Default MC** (drives recommendations; must feel responsive): target
     **~120 samples** on moderate positions, ~2-3s wall-clock with the pool.
   - **Deep Analyze** (explicit, on demand): target **~800 samples**
     (≈6-8x default) AND a higher per-world `solve_budget` (e.g. 150k-200k vs
     the default 30k-60k), ~15-20s budget.
   - Keep the adaptive `_choose_effort` behavior: exhaustive enumeration when
     the arrangement space is tiny (exact, no sampling), sampling otherwise.
     The tier sizes above are the *sampling* targets, capped by time.

## Work breakdown

### Phase 1 — Parallel world evaluation (biggest lever)

- Add a module-level, lazily-created, reused `ProcessPoolExecutor`
  (`concurrent.futures`) sized to `min(os.cpu_count(), 12)`. Reuse across
  requests (create once, keep a global handle); add a clean shutdown hook.
- Refactor the per-world work into a **top-level, picklable worker function**
  (process pools can't pickle closures/locals). The worker receives a
  serialized `GameState` (or the args to rebuild one) plus the list of moves to
  test and the `solve_budget`, and returns, for that single world: which moves
  win (bool per move) and whether the world is winnable at all. Keep workers
  pure (no shared mutable state).
- Parallelize BOTH `estimate` (per-move) and `position_estimate` (position
  only). Dispatch worlds in batches to amortize IPC; aggregate results in the
  parent. Preserve the fair-comparison property: every move is tested against
  the SAME set of completed worlds (drop any partially-returned batch at the
  time cap rather than counting it, to keep `wins <= trials`).
- Wall-clock budget: stop submitting new work once the deadline passes; collect
  already-finished results. Set `timedOut=True` if the target sample count
  wasn't reached.
- **Single-process fallback**: if pool creation fails or `cpu_count()==1`, use
  the existing sequential loop. Wrap worker dispatch so a worker exception
  degrades to sequential rather than failing the request.
- Verify `GameState`/`Card`/`Move` pickle round-trip in a test before relying
  on it.

### Phase 2 — Racing / early elimination (reduce wasted samples)

- In `estimate` (per-move comparison), after each batch of worlds, compute each
  move's Wilson interval. **Eliminate** a move from further sampling once its
  upper confidence bound is below the current leader's lower confidence bound
  (it cannot catch up). Stop early when only one contender remains or the time
  budget ends.
- This must not change the reported `perMove` for eliminated moves beyond
  marking them settled; keep their last-known winRate + margin and a flag like
  `eliminated: true`. The `best` move is the surviving leader.
- Keep the fair-comparison invariant for the *contending* moves (compare on the
  same completed-world set).

### Phase 3 — Deep Analyze tier + app wiring

- Add constants in `app.py`:
  - `MC_SAMPLES = 120`, `MC_SOLVE_BUDGET` ~ 60_000, `MC_TIME_BUDGET_S` ~ 4-5s
    for the DEFAULT path (tune so moderate positions finish in ~2-3s parallel).
  - `MC_DEEP_SAMPLES = 800`, `MC_DEEP_SOLVE_BUDGET = 200_000`,
    `MC_DEEP_TIME_BUDGET_S = 20.0`.
- `/api/analyze/<id>`: accept a `{deep: true}` flag (and/or explicit
  `samples/budget/timeBudget` overrides). When `deep`, use the Deep tier.
- UI (`static/index.html`, `app.js`, `styles.css`): add a **"Deep analyze"**
  button next to "Analyze (Monte Carlo)". It calls `/api/analyze` with
  `deep:true`, shows the busy indicator (the existing `setBusy`), and renders
  the result via the existing `renderMcResult` (which already shows the
  confidence interval). Reuse the existing patterns; do not invent new UI
  conventions.
- Optional seed: thread an optional `seed` through `/api/analyze`,
  `/api/winprob`, and the MC functions for reproducibility.

### Phase 4 — Tests & verification

- Add sanity checks to `check.py` (follow the existing style — standalone
  functions called from `__main__`, using `build_opaque_state`/`parse_midgame`
  or hand-built `GameState`s):
  - Pickle round-trip of `GameState`/`Card`/`Move`.
  - Parallel `estimate` agrees with sequential `estimate` on a fixed seed /
    exhaustive (exact) position (same `winProbability` and same `best`).
  - `wins <= trials` holds for every move after a time-limited parallel run.
  - Racing elimination never eliminates the eventual best move on an exact
    (exhaustive) position where one move is strictly dominant.
  - Existing checks still pass: dead-stock game over (greedy AND MC), exact win
    plan, forced-move pruning, unsafe-foundation-beats-dead-draw, Wilson
    interval degenerate inputs.
- Measure and report wall-clock: sequential vs parallel on a heavy position
  (~17-21 hidden cards) at equal sample counts, and the sample count reachable
  in the default time budget before vs after. Target: default path reaches
  ~120 samples in ~2-3s; Deep reaches ~800.
- Confirm no Flask request hangs; pool reused across requests (not recreated
  per call).

## Risks / watch-outs

- **GIL**: must use processes, not threads (solver is pure-Python CPU-bound).
- **Pickling**: closures/lambdas can't cross process boundaries — worker must be
  a top-level function; args must be picklable.
- **Pool lifecycle in Flask dev server + debug reloader**: the reloader spawns a
  child process; create the pool lazily on first use, guard against double
  creation, and don't create it at import time. Ensure `if __name__` main guard
  for multiprocessing on macOS (spawn start method) — worker module must be
  importable without side effects.
- **macOS spawn**: default start method is "spawn"; the worker function and its
  module imports must be importable cleanly (no heavy import-time work).
- **Determinism with parallelism**: exact reproducibility across processes is
  hard; aim for "stable recommendation," and for the seeded test use the
  sequential or exhaustive path for exact equality.
- **Don't regress responsiveness**: the DEFAULT path must still feel quick;
  if parallel startup overhead dominates for tiny positions, keep using the
  sequential/exhaustive path for small hidden-card counts (reuse `_choose_effort`).

## Deliverables / report back to the orchestrating agent

- Summary of changes per file.
- Before/after timings (sequential vs parallel; samples reached in default
  budget) on a heavy position.
- Confirmation all `check.py` sanity checks pass (paste the tail).
- Any invariant that could not be preserved, and why.
- Note whether `GameState` was picklable as-is or needed changes.

## How to run

```
.venv/bin/python check.py          # sanity suite
.venv/bin/python app.py            # dev server on :5001
```
Clear test rows after manual testing: `DELETE FROM games;` in `games.db`.
