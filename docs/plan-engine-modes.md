# Plan: Engine mode selector + confidence short-circuit + adaptive Deep MC

## Context

Klondike solitaire ASSISTANT (Python/Flask + vanilla JS). It recommends moves
for a mid-game position with hidden (face-down) cards. Three recommenders exist:
- `solver/strategy.py::plan_steps` — fast greedy planner (no sampling).
- `solver/montecarlo.py::mc_plan_steps` / `estimate` / `position_estimate` —
  Monte Carlo over sampled hidden-card arrangements, parallelized across a
  process pool (`solver/parallel.py`), with racing elimination and 95% Wilson
  confidence intervals. Logs to `logs/mc.log` via `solver/mclog.py`.

Current app wiring (`app.py`):
- `Game` dataclass has `mc_mode: bool`. `_compute_plan(game, n)` routes:
  exact_win_plan (proven win) → mc_plan_steps (if `mc_mode`) → `plan_steps`
  (greedy).
- Tiers (module constants): default `MC_SAMPLES=120 / MC_SOLVE_BUDGET=60_000 /
  MC_TIME_BUDGET_S=5.0`; deep `MC_DEEP_SAMPLES=800 / MC_DEEP_SOLVE_BUDGET=200_000
  / MC_DEEP_TIME_BUDGET_S=20.0`.
- Endpoints: `/api/mcmode/<id>` toggles `mc_mode`; `/api/analyze/<id>` runs a
  one-off per-move analysis (`{deep:true}` picks the deep tier) but does NOT
  change the stepping recommendation; `/api/winprob/<id>` is the cheap
  position estimate used for the readout.
- UI (`static/index.html`, `app.js`, `styles.css`): a "MC drives
  recommendations" checkbox (calls `/api/mcmode`), an "Analyze (Monte Carlo)"
  and a "Deep analyze" button (call `/api/analyze`).

### Why we're changing it (from real logs of two lost games)

- The toggle only ever drove recommendations at the FAST tier (5s/120). On
  heavy positions that completed only ~15-30 of 120 samples (wide ±0.17-0.23
  intervals), so midgame moves in the murky ~0.4-0.6 band were under-sampled.
- "Deep analyze" is a disconnected one-off readout; it never fed the stepper.
- Users found this confusing ("do I have to Deep Analyze every move?").

## Product decisions (confirmed)

1. **Engine chosen once at setup** (on the New-game screen), three options:
   - **Normal strategy** — greedy planner, instant.
   - **Fast MC** — Monte Carlo, caps at the FAST tier when a move is ambiguous.
   - **Deep MC** — Monte Carlo, ADAPTIVE: escalates to the DEEP tier only when a
     move is still ambiguous after a fast pass.
2. **Confidence short-circuit for BOTH MC tiers**: stop sampling early and
   return the fast recommendation as soon as the answer is confident (proven
   exact, or the leading move is statistically separated from the runner-up).
3. **Deep MC is adaptive**: fast when the position is clear, deep only when
   ambiguous. (So Deep differs from Fast only in how hard it thinks on genuinely
   close moves.)
4. Keep the manual **Deep Analyze button** as a per-position override.

## Behavior spec

| Mode | Clear / settled position | Ambiguous position |
|---|---|---|
| Normal | instant greedy | instant greedy |
| Fast MC | short-circuits (~1-2s) | up to FAST tier (5s / 120 samples) |
| Deep MC | short-circuits (~1-2s) | escalates to DEEP tier (20s / 800 samples) |

**"Confident / settled" = any of:**
- The position is a proven win (`exact_win_plan` fires) — already handled,
  return immediately.
- Exhaustive/exact enumeration (tiny hidden-card count) — already exact.
- **Statistically separated leader**: after the fast pass, the best move's
  Wilson LOWER bound is above the second-best move's Wilson UPPER bound (the
  racing-elimination signal already computed in `montecarlo`), OR
- Win probability is effectively decided (near 0 or near 1 within the margin).

**Adaptive Deep MC algorithm (per recommendation):**
1. Run a FAST pass (FAST tier budget).
2. If settled by the criteria above → return the fast result (short-circuit).
3. Else continue sampling up to the DEEP tier budget (reuse the same evaluated
   worlds; don't throw away the fast pass's work — extend it). Return when
   settled or the deep budget/time is exhausted.

**Fast MC** uses the same short-circuit but never escalates past the FAST tier.

Implementation note: `estimate`/`position_estimate` already sample
world-by-world with a wall-clock deadline and track per-move Wilson intervals
and racing elimination. The cleanest implementation is to add an early-stop
check inside the sampling loop ("settled?") and, for Deep, a two-phase budget
(fast target/time first, then extend to deep) rather than two separate calls
that re-sample from scratch. Keep the `wins <= evaluated` invariant and the
fair-comparison property (compare moves on the same completed-world set).

## Hard invariants (do NOT break)

1. All existing `check.py` sanity checks pass, plus new ones you add.
   (`.venv/bin/python check.py` ends "All sanity checks passed.")
2. `estimate`/`position_estimate` keep returning the same dict shape/fields
   (`winProbability`, `winMargin`, `exact`, `mode`, `samples`, `timedOut`,
   `perMove`, `best`, `gameOver`, plus `forced`/`eliminated` where present).
   Add fields if needed (e.g. `settled`, `escalated`) but don't remove any.
3. `wins <= evaluated` for every move always (a prior crash came from wins >
   trials → complex sqrt in the Wilson interval). Short-circuit/escalation must
   not violate this.
4. Proven-win shortcut (`exact_win_plan`) and dead-stock-cycle game-over
   detection must still fire and take precedence.
5. `GameState` stays picklable (process pool depends on it). No shared mutable
   global caches across worlds/processes; keep per-call locals.
6. Graceful fallback: pool/worker failure degrades to sequential; never hang or
   crash the Flask request.
7. Parallelism stays on for heavy positions; short-circuit must work in BOTH
   the parallel and sequential paths.
8. Determinism option: the optional RNG seed still produces reproducible
   results where applicable.

## Work breakdown

### Backend — montecarlo.py
- Add a `settled(...)` helper: given current per-move win counts + evaluated
  count, returns True when the leader is statistically separated from the
  runner-up (Wilson LOWER(best) > Wilson UPPER(2nd)), or win prob is decided
  (<= margin from 0 or 1), or only one contender remains (racing). Reuse
  `_wilson_interval` and the existing racing logic.
- Add early-stop to the sampling loops (parallel `_estimate_core_par` /
  sequential `_estimate_core_seq`, and the position-estimate equivalents): after
  each batch, if `settled(...)`, stop and return what we have, flagged
  `settled=True`.
- Add a two-phase "adaptive" entry point, e.g.
  `estimate_adaptive(state, fast_tier, deep_tier, ...)` (or a `deep: bool` +
  `adaptive: bool` param on `estimate`): run the fast target first; if not
  settled, EXTEND sampling (same worlds + more) to the deep target. Expose a
  matching `mc_plan_steps_adaptive` (or extend `mc_plan_steps`) returning the
  usual plan dict plus `escalated: bool` and `settled: bool`.
- Keep `estimate`/`position_estimate` working as today for the one-off
  `/api/analyze` and `/api/winprob` paths.

### Backend — app.py
- Replace `Game.mc_mode: bool` with `Game.engine: str` in
  {"greedy", "fast_mc", "deep_mc"} (default "greedy"). Keep backward-compat if
  trivial, else update all readers.
- `/api/new` reads `{engine: "...", ...}` from the body and stores it on the
  game. Default "greedy" if absent.
- `_compute_plan(game, n)`:
  - proven win → exact_win_plan (unchanged, highest precedence).
  - engine == "greedy" → `plan_steps`.
  - engine == "fast_mc" → MC with FAST tier + short-circuit.
  - engine == "deep_mc" → MC adaptive (fast pass, escalate to DEEP if not
    settled) + short-circuit.
- Replace `/api/mcmode` with `/api/engine/<id>` (set engine mid-game) — OR keep
  the endpoint name and accept `{engine: "..."}`; pick one and update the UI.
  Recompute the current recommendation under the new engine and return it.
- `_plan_response` should surface the mode used and any `settled`/`escalated`
  flags so the UI can show them.
- Keep `/api/analyze` (one-off) and `/api/winprob` as-is.
- Log via `mclog`: engine chosen per game; per recommendation whether it
  short-circuited (settled) or escalated (deep), evaluated/target, elapsed. This
  is important for validating the behavior from `logs/mc.log`.

### Frontend — static/
- On the setup screen (`index.html`), add an **engine selector** (radio or
  dropdown): Normal strategy / Fast MC / Deep MC. Default Normal (or Fast MC —
  pick one; Normal is safest default). `app.js` `collectLayout()`-equivalent
  sends `engine` in the `/api/new` body.
- Remove the old "MC drives recommendations" checkbox (its behavior is now the
  engine selector). Keep an in-game way to CHANGE engine (reuse the selector,
  wired to the set-engine endpoint) — optional but nice.
- Keep "Analyze (Monte Carlo)" and "Deep analyze" buttons (manual overrides).
- The win-probability readout (`renderMcResult`) should indicate when a result
  was settled early vs escalated to deep (small tag, reuse existing patterns).
- Respect the existing no-cache headers / setBusy busy indicator / samples
  field (don't reintroduce the "undefined sampled layouts" bug — ensure the
  plan response still carries `samples`).

## Validation / report back

Use `.venv/bin/python`. Do NOT commit; leave changes unstaged.

1. `.venv/bin/python check.py` → "All sanity checks passed." Add checks:
   - `settled()` logic: a clearly-separated leader returns True; a near-tie
     returns False.
   - Short-circuit: on an exact/clear position, MC stops early (fewer than the
     full target samples) and still returns the correct best move.
   - Adaptive Deep: on a CLEAR position it does NOT escalate (stays ~fast); on
     an AMBIGUOUS position it DOES escalate (samples beyond the fast target).
     Build deterministic fixtures (seeded) for both.
   - `wins <= evaluated` still holds after short-circuit/escalation.
   - Proven-win and dead-stock-cycle still reported correctly through each
     engine mode.
2. End-to-end via the server (`.venv/bin/python app.py`): create a game with
   each engine; confirm `/api/new` stores it and `_compute_plan` routes
   correctly (check `logs/mc.log` lines). For Deep MC, show a CLEAR position
   returning fast (short-circuit, elapsed ~1-2s) and an AMBIGUOUS position
   escalating (elapsed up toward 20s, samples beyond fast target).
3. Timings: for a heavy ~20-hidden position, report Deep-MC elapsed on a settled
   vs an ambiguous move, and the samples evaluated in each.

## Risks / watch-outs

- Don't re-sample from scratch when escalating — extend the fast pass's worlds,
  or you waste the budget and slow every deep move.
- Short-circuit must not fire TOO eagerly (e.g. after 2 samples) — require a
  minimum number of evaluated worlds before trusting separation (reuse the
  racing guard, which already needs ~8 worlds).
- Keep the fair-comparison invariant: moves compared on the same completed-world
  set; dropping an unfinished batch at the deadline must not leave wins >
  evaluated.
- Don't reintroduce the "samples undefined in MC-drive response" bug: the plan
  response for the stepping path must include `samples` (and now `settled` /
  `escalated`).
- macOS spawn: workers re-import modules; no import-time side effects; don't log
  the startup banner from workers (already guarded in app.py main-process check).

## Deliverables (report back)

- Per-file change summary.
- check.py tail (all pass, incl. new checks).
- Evidence from `logs/mc.log`: Deep MC short-circuiting on a clear move
  (fast/settled) and escalating on an ambiguous move (deep/escalated), with
  elapsed + samples for each.
- Confirmation the three engine modes route correctly and the stepping
  recommendation now reflects the chosen engine.
- Any invariant not preserved and why. Do NOT commit.

## How to run
```
.venv/bin/python check.py
.venv/bin/python app.py      # :5001
```
Logs: `logs/mc.log`. Clear test rows: `DELETE FROM games;` in `games.db`.
