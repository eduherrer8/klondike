"""Monte Carlo determinization for play under uncertainty.

The strategy planner (strategy.py) reasons only about known cards and plays a
greedy, shallow-lookahead line. Monte Carlo is stronger: it samples many
concrete arrangements of the hidden (face-down) cards that are consistent with
everything currently visible, runs the exact solver on each, and recommends the
move that wins across the most sampled worlds. It also yields an estimated win
probability from the current position.

Because this game's stock is fully known, the ONLY uncertainty is the order of
the face-down tableau cards. The identities of those cards are fixed by
elimination (deck minus everything with a known location); only their placement
into the face-down slots varies between samples.
"""

from __future__ import annotations

import random

from .cards import UNKNOWN, Card, full_deck, is_unknown
from .engine import apply_move, legal_moves
from .game import GameState, Move
from .parallel import (
    eval_world_moves,
    eval_world_winnable,
    get_pool,
    pool_size,
)
from .solver import is_winnable, solve

# How many worlds to dispatch per parallel batch, as a multiple of the pool
# size. One wave per batch keeps the wall-clock deadline responsive: once the
# deadline passes we stop submitting, so the worst-case overrun is a single
# in-flight wave rather than a big backlog. Racing elimination also re-evaluates
# between batches, so smaller batches react faster.
_BATCH_MULTIPLIER = 1


def _hidden_slots(state: GameState) -> list[tuple[int, int]]:
    """Positions (col, index) of every face-down unknown card in the tableau."""
    slots = []
    for ci, col in enumerate(state.tableau):
        for pi, (card, up) in enumerate(col):
            if is_unknown(card) and not up:
                slots.append((ci, pi))
    return slots


def _unknown_pool(state: GameState) -> list[Card]:
    """Cards whose identity is still unknown = full deck minus every card with
    a known location (concrete tableau cards, waste, stock, foundations)."""
    known: set[Card] = set()
    for col in state.tableau:
        for card, _ in col:
            if not is_unknown(card):
                known.add(card)
    known.update(state.waste)
    known.update(state.stock)
    for pile in state.foundations.values():
        known.update(pile)
    return [c for c in full_deck() if c not in known]


def _build_world(state: GameState, slots: list[tuple[int, int]],
                 perm) -> GameState:
    """Clone `state` and assign the given card permutation to the face-down
    `slots` (identity becomes concrete; cards stay face-down)."""
    s = state.clone()
    for (ci, pi), card in zip(slots, perm):
        col = s.tableau[ci]
        _, up = col[pi]
        col[pi] = (card, up)
    return s


def _determinize(state: GameState, slots: list[tuple[int, int]],
                 pool: list[Card], rng: random.Random) -> GameState:
    """A random determinized world: a random permutation of `pool` into slots."""
    perm = pool[:]
    rng.shuffle(perm)
    return _build_world(state, slots, perm)


def _wilson_interval(wins: int, trials: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score 95% confidence interval for a binomial proportion.

    Returns (p_hat, half_width, margin) where p_hat is wins/trials and margin
    is the symmetric half-width of the interval (so the CI is p_hat +/- margin,
    clamped to [0, 1]). More reliable than the normal approximation for small
    samples and extreme proportions."""
    if trials <= 0:
        return (0.0, 1.0, 1.0)
    # Clamp p into [0, 1] defensively so the variance term can't go negative
    # (which would make the sqrt complex) from any upstream counting glitch.
    p = min(1.0, max(0.0, wins / trials))
    z2 = z * z
    denom = 1 + z2 / trials
    centre = (p + z2 / (2 * trials)) / denom
    variance = p * (1 - p) / trials + z2 / (4 * trials * trials)
    half = (z * (max(0.0, variance) ** 0.5)) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (p, (hi - lo) / 2, half)


def _choose_effort(num_hidden: int, requested_samples: int) -> dict:
    """Adaptive effort selection based on how many cards are still hidden.

    - Few hidden cards -> the arrangement space is tiny; enumerate it exactly
      (exhaustive) for an exact probability.
    - Moderate -> the classic ~40-80 determinization sweet spot.
    - Many -> cap sampling and lean on the wall-clock budget; the estimate is
      necessarily rough but the move RANKING is usually still reliable.
    """
    import math

    # Distinct arrangements = num_hidden! (cards into ordered slots). Enumerate
    # exhaustively only when that's small enough to be cheap.
    if num_hidden <= 1:
        arrangements = 1
    else:
        arrangements = math.factorial(num_hidden) if num_hidden <= 7 else None

    if arrangements is not None and arrangements <= 120:
        return {"mode": "exhaustive", "samples": arrangements, "arrangements": arrangements}
    # Sampled modes: honor the requested target (the caller's tier: ~120
    # default, ~800 deep). The wall-clock deadline is the real governor now —
    # it stops sampling early if the target can't be reached — so we no longer
    # hard-cap the count here (which silently throttled the deep tier). Keep a
    # sensible floor so a tiny request still samples enough to rank moves.
    return {"mode": "sampled", "samples": max(40, requested_samples)}


def _world_stream(state: GameState, slots, pool, consistent, mode, rng):
    """Yield determinized worlds per the chosen effort mode."""
    import itertools as _it

    if not slots or not consistent:
        yield state.clone()
    elif mode == "exhaustive":
        for perm in _it.permutations(pool):
            yield _build_world(state, slots, perm)
    else:
        while True:
            yield _determinize(state, slots, pool, rng)


def _should_parallelize(num_hidden: int, mode: str, planned: int) -> bool:
    """Whether the process pool is worth the IPC/startup overhead.

    Tiny positions (few hidden cards, exhaustive enumeration, or a handful of
    planned worlds) finish faster in-process than the cost of shipping worlds to
    workers, so keep those sequential (this also protects the DEFAULT path's
    responsiveness on light positions). Reuse the effort signal already chosen
    by `_choose_effort`.
    """
    if pool_size() <= 1:
        return False
    if mode in ("exhaustive", "exact"):
        return False
    if num_hidden <= 7:
        return False
    # Only parallelize when there's enough work to spread across workers.
    return planned >= 16


def _submit_batch(pool, worlds, fn, extra):
    """Submit one batch of worlds to `fn` and return the list of futures. The
    world is always the first positional arg; `extra` is appended after it."""
    return [pool.submit(fn, w, *extra) for w in worlds]


def _drain_batch(futures, deadline, _time):
    """Yield completed future results, honoring the wall-clock deadline.

    Waits for futures as they finish; once the deadline passes it stops waiting,
    cancels whatever is still pending, and ignores their results. Each future
    corresponds to one world that is evaluated ATOMICALLY (all moves for that
    world run inside the single worker call), so dropping an unfinished future
    just drops that whole world — keeping `wins <= evaluated` and fair
    comparison across the counted worlds.

    Yields (result, timed_out_flag) per completed future; the flag is only ever
    True on the sentinel final yield when the deadline cut collection short.
    """
    from concurrent.futures import FIRST_COMPLETED, wait

    pending = set(futures)
    while pending:
        if deadline is None:
            timeout = None
        else:
            timeout = deadline - _time.monotonic()
            if timeout <= 0:
                for f in pending:
                    f.cancel()
                yield (None, True)
                return
        done, pending = wait(pending, timeout=timeout,
                              return_when=FIRST_COMPLETED)
        if not done:
            # Timed out waiting: cancel the rest and signal the cut.
            for f in pending:
                f.cancel()
            yield (None, True)
            return
        for f in done:
            yield (f.result(), False)


def position_estimate(state: GameState, samples: int = 60,
                      solve_budget: int = 60_000,
                      rng: random.Random | None = None,
                      time_budget_s: float = 8.0) -> dict:
    """CHEAP win-probability for the CURRENT position.

    Unlike `estimate`, this does NOT evaluate every legal move. It samples the
    position once per world and asks "is this world winnable at all?" — one
    solve per world instead of (moves x worlds). Use this for the
    win-probability readout; use `estimate` only when you need to compare moves.

    Sampling across worlds is run on a shared process pool when the position is
    heavy enough to benefit; it falls back cleanly to a single-process loop for
    light positions or if the pool is unavailable.
    """
    import time as _time

    rng = rng or random.Random()
    deadline = (_time.monotonic() + time_budget_s) if time_budget_s else None

    if state.is_won():
        return {"winProbability": 1.0, "winMargin": 0.0, "exact": True,
                "mode": "won", "samples": 0, "won": True, "gameOver": False,
                "timedOut": False}
    if not legal_moves(state):
        return {"winProbability": 0.0, "winMargin": 0.0, "exact": True,
                "mode": "gameover", "samples": 0, "won": False,
                "gameOver": True, "timedOut": False}
    from .strategy import is_dead_stock_cycle
    if is_dead_stock_cycle(state):
        return {"winProbability": 0.0, "winMargin": 0.0, "exact": True,
                "mode": "gameover", "samples": 0, "won": False,
                "gameOver": True, "stuck": True, "timedOut": False}

    slots = _hidden_slots(state)
    pool = _unknown_pool(state)
    consistent = (len(pool) == len(slots))
    num_hidden = len(slots) if consistent else 0

    effort = _choose_effort(num_hidden, samples)
    mode = effort["mode"] if (num_hidden and consistent) else "exact"
    planned = effort["samples"] if (num_hidden and consistent) else 1

    from .mclog import log as _log
    _pt0 = _time.monotonic()
    _par = _should_parallelize(num_hidden, mode, planned)
    _log(f"position_estimate START hidden={num_hidden} mode={mode} "
         f"planned={planned} parallel={_par} pool={pool_size()}")

    winnable, evaluated, timed_out = _position_eval(
        state, slots, pool, consistent, mode, planned, solve_budget, rng,
        deadline, _time)

    exact = (mode in ("exhaustive", "exact")) and not timed_out
    p, margin, _ = _wilson_interval(winnable, evaluated)
    _log(f"position_estimate DONE winnable={winnable}/{evaluated} "
         f"winProb={(winnable/evaluated) if evaluated else 0.0:.2f} "
         f"timedOut={timed_out} elapsed={_time.monotonic()-_pt0:.1f}s")
    return {
        "winProbability": (winnable / evaluated) if evaluated else 0.0,
        "winMargin": 0.0 if exact else margin,
        "exact": exact,
        "mode": mode,
        "samples": evaluated,
        "won": False,
        "gameOver": False,
        "timedOut": timed_out,
    }


def _position_eval(state, slots, pool_cards, consistent, mode, planned,
                   solve_budget, rng, deadline, _time):
    """Evaluate `planned` worlds for 'winnable at all?', parallel or sequential.

    Returns (winnable, evaluated, timed_out). Only fully-completed worlds are
    counted so `winnable <= evaluated` always holds.
    """
    executor = get_pool() if _should_parallelize(
        len(slots) if consistent else 0, mode, planned) else None

    if executor is None:
        return _position_eval_seq(state, slots, pool_cards, consistent, mode,
                                  planned, solve_budget, rng, deadline, _time)
    try:
        return _position_eval_par(executor, state, slots, pool_cards, consistent,
                                  mode, planned, solve_budget, rng, deadline,
                                  _time)
    except Exception:
        # A worker/pool failure must never break the request: fall back to the
        # in-process path for a clean (if slower) answer.
        return _position_eval_seq(state, slots, pool_cards, consistent, mode,
                                  planned, solve_budget, rng, deadline, _time)


def _position_eval_seq(state, slots, pool_cards, consistent, mode, planned,
                       solve_budget, rng, deadline, _time):
    winnable = 0
    evaluated = 0
    timed_out = False
    for world in _world_stream(state, slots, pool_cards, consistent, mode, rng):
        if evaluated >= planned:
            break
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        if is_winnable(world, max_states=solve_budget):
            winnable += 1
        evaluated += 1
    return winnable, evaluated, timed_out


def _position_eval_par(executor, state, slots, pool_cards, consistent, mode,
                       planned, solve_budget, rng, deadline, _time):
    winnable = 0
    evaluated = 0
    timed_out = False
    batch = max(pool_size() * _BATCH_MULTIPLIER, 1)
    stream = _world_stream(state, slots, pool_cards, consistent, mode, rng)

    while evaluated < planned:
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        want = min(batch, planned - evaluated)
        worlds = []
        for _ in range(want):
            try:
                worlds.append(next(stream))
            except StopIteration:
                break
        if not worlds:
            break
        futures = _submit_batch(executor, worlds, eval_world_winnable,
                                (solve_budget,))
        for result, cut in _drain_batch(futures, deadline, _time):
            if cut:
                timed_out = True
                break
            if result:
                winnable += 1
            evaluated += 1
        if timed_out:
            break
    return winnable, evaluated, timed_out


def exact_win_plan(state: GameState, n: int = 3, solve_budget: int = 200_000):
    """If the position is a PROVEN win, return a guaranteed plan.

    Returns a plan dict (same shape as strategy.plan_steps, with
    exactGuaranteed=True) when the win is certain, else None.

    Two cases:
      * No hidden cards: the exact solver yields one guaranteed winning line;
        follow it.
      * Hidden cards but FEW enough to enumerate, and EVERY arrangement is
        winnable: pick a next move that wins in all arrangements and stop at the
        next reveal (beyond a flip the line depends on the real card).
    """
    import itertools as _it

    from .strategy import flips_hidden_card

    if state.is_won() or not legal_moves(state):
        return None

    slots = _hidden_slots(state)
    pool = _unknown_pool(state)
    consistent = (len(pool) == len(slots))

    def _steps_from_line(line):
        steps = []
        sim = state
        reveal_next = False
        for i, mv in enumerate(line[:n]):
            rv = flips_hidden_card(sim, mv)
            steps.append({
                "index": i, "kind": mv.kind, "description": mv.describe(),
                "revealsHidden": rv,
                "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                         "count": mv.count, "card": mv.card},
            })
            if i == 0:
                reveal_next = rv
            if rv:
                break
            sim = apply_move(sim, mv)
        return steps, reveal_next

    # Case 1: fully known -> single guaranteed line.
    if not slots:
        line = solve(state, max_states=solve_budget)
        if line is None:
            return None
        steps, reveal_next = _steps_from_line(line)
        return {"solved": False, "gameOver": False, "stuck": False,
                "steps": steps, "revealNext": reveal_next,
                "winProbability": 1.0, "winMargin": 0.0, "exact": True,
                "exactGuaranteed": True, "mc": False}

    # Case 2: hidden cards, but only if enumerable (keeps this cheap & certain).
    import math
    if not consistent or len(slots) > 7 or math.factorial(len(slots)) > 120:
        return None

    worlds = [_build_world(state, slots, perm) for perm in _it.permutations(pool)]
    # Every arrangement must be winnable for a guarantee.
    if any(solve(w, max_states=solve_budget) is None for w in worlds):
        return None

    # Find a first move that wins in EVERY arrangement.
    for mv in legal_moves(state):
        if all(solve(apply_move(w, mv), max_states=solve_budget) is not None
               for w in worlds):
            steps, reveal_next = _steps_from_line([mv])
            # Extend with a continuation from one world, up to the next reveal.
            if not reveal_next:
                after = apply_move(state, mv)
                cont = exact_win_plan(after, n=n - 1, solve_budget=solve_budget)
                if cont and cont["steps"]:
                    steps = steps + [
                        {**s, "index": len(steps) + j}
                        for j, s in enumerate(cont["steps"])
                    ]
            return {"solved": False, "gameOver": False, "stuck": False,
                    "steps": steps[:n], "revealNext": reveal_next,
                    "winProbability": 1.0, "winMargin": 0.0, "exact": True,
                    "exactGuaranteed": True, "mc": False}
    return None


def estimate(state: GameState, samples: int = 60, solve_budget: int = 60_000,
             rng: random.Random | None = None,
             time_budget_s: float = 8.0) -> dict:
    """Monte Carlo analysis of the current position.

    Returns a dict:
      - winProbability: fraction of sampled worlds winnable with best play
      - samples: number of worlds actually evaluated
      - perMove: list of {move, winRate, wins, trials, description} for each
        legal move, sorted best-first
      - best: the recommended move dict (highest win rate), or None
      - gameOver: True if there are no legal moves at all
      - timedOut: True if the wall-clock budget was hit (result is best-effort)

    `time_budget_s` caps the wall-clock time so MC never freezes the UI on
    positions with many hidden cards; it returns the best estimate gathered so
    far. Set to 0 to disable the cap.
    """
    import time as _time

    rng = rng or random.Random()
    deadline = (_time.monotonic() + time_budget_s) if time_budget_s else None

    if state.is_won():
        return {"winProbability": 1.0, "samples": 0, "perMove": [],
                "best": None, "won": True, "gameOver": False}

    moves = legal_moves(state)
    if not moves:
        return {"winProbability": 0.0, "samples": 0, "perMove": [],
                "best": None, "won": False, "gameOver": True}

    # Dead stock cycle: the only moves left are stock draws that never expose a
    # playable card. That's a game over even though a 'draw' is technically
    # legal, so MC must report it the same way the greedy planner does.
    from .strategy import is_dead_stock_cycle
    if is_dead_stock_cycle(state):
        return {"winProbability": 0.0, "winMargin": 0.0, "exact": True,
                "mode": "gameover", "samples": 0, "perMove": [],
                "best": None, "won": False, "gameOver": True, "stuck": True,
                "timedOut": False}

    # Forced-move pruning: with a single legal move there's no choice to
    # compare, so skip the expensive per-move loop and just report the cheap
    # position estimate for that one move.
    if len(moves) == 1:
        pos = position_estimate(state, samples=samples, solve_budget=solve_budget,
                                rng=rng, time_budget_s=time_budget_s)
        mv = moves[0]
        per_move = [{
            "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                     "count": mv.count, "card": mv.card},
            "description": mv.describe(),
            "wins": round(pos["winProbability"] * pos["samples"]),
            "trials": pos["samples"],
            "winRate": pos["winProbability"],
            "margin": pos["winMargin"],
        }]
        return {
            "winProbability": pos["winProbability"],
            "winMargin": pos["winMargin"],
            "exact": pos["exact"],
            "mode": pos["mode"],
            "samples": pos["samples"],
            "consistent": True,
            "timedOut": pos["timedOut"],
            "perMove": per_move,
            "best": per_move[0]["move"],
            "forced": True,
            "won": False,
            "gameOver": False,
        }

    slots = _hidden_slots(state)
    pool = _unknown_pool(state)
    consistent = (len(pool) == len(slots))
    num_hidden = len(slots) if consistent else 0

    # Adaptive effort: enumerate exactly when the arrangement space is tiny,
    # otherwise sample in the accepted ~40-80 range (capped for heavy cases).
    effort = _choose_effort(num_hidden, samples)
    mode = effort["mode"]
    planned = effort["samples"]
    if num_hidden == 0 or not consistent:
        planned = 1
        mode = "exact"  # fully known position: a single deterministic world

    from .mclog import log as _log
    _t0 = _time.monotonic()
    parallel = _should_parallelize(num_hidden, mode, planned)
    _log(f"estimate START moves={len(moves)} hidden={num_hidden} mode={mode} "
         f"planned={planned} parallel={parallel} pool={pool_size()} "
         f"solve_budget={solve_budget} time_budget={time_budget_s}")

    agg = _estimate_core(state, moves, slots, pool, consistent, mode, planned,
                         solve_budget, rng, deadline, _time)
    wins = agg["wins"]
    evaluated = agg["evaluated"]
    world_any_win_count = agg["world_any_win_count"]
    timed_out = agg["timed_out"]
    eliminated = agg["eliminated"]

    # Exhaustive + no timeout = exact (no sampling error).
    exact = (mode in ("exhaustive", "exact")) and not timed_out

    per_move = []
    for mi, mv in enumerate(moves):
        p, margin, _ = _wilson_interval(wins[mi], evaluated)
        per_move.append({
            "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                     "count": mv.count, "card": mv.card},
            "description": mv.describe(),
            "wins": wins[mi],
            "trials": evaluated,
            "winRate": (wins[mi] / evaluated) if evaluated else 0.0,
            "margin": 0.0 if exact else margin,
            "eliminated": mi in eliminated,
        })

    # Surviving (non-eliminated) moves rank first, then by win rate. An
    # eliminated move is proven to trail the leader, so it can never be `best`.
    per_move.sort(key=lambda m: (not m["eliminated"], m["winRate"]), reverse=True)
    win_probability = (world_any_win_count / evaluated) if evaluated else 0.0
    _, wp_margin, _ = _wilson_interval(world_any_win_count, evaluated)
    best = per_move[0]["move"] if per_move else None

    _best = per_move[0] if per_move else None
    _log(f"estimate DONE evaluated={evaluated}/{planned} "
         f"winProb={win_probability:.2f}+/-{(0.0 if exact else wp_margin):.2f} "
         f"timedOut={timed_out} eliminated={len(eliminated)}/{len(moves)} "
         f"best={(_best['description'] + ' @%.2f' % _best['winRate']) if _best else None} "
         f"elapsed={_time.monotonic() - _t0:.1f}s")

    return {
        "winProbability": win_probability,
        "winMargin": 0.0 if exact else wp_margin,
        "exact": exact,
        "mode": mode,
        "samples": evaluated,
        "consistent": consistent,
        "timedOut": timed_out,
        "perMove": per_move,
        "best": best,
        "won": False,
        "gameOver": False,
    }


def _estimate_core(state, moves, slots, pool_cards, consistent, mode, planned,
                   solve_budget, rng, deadline, _time):
    """Evaluate `planned` worlds across all `moves`, returning aggregate win
    counts. Dispatches to the parallel path when worthwhile, else sequential.
    Guarantees ``wins[mi] <= evaluated`` for every move (fair comparison), and
    applies racing elimination to stop sampling moves that cannot catch up.
    """
    executor = get_pool() if _should_parallelize(
        len(slots) if consistent else 0, mode, planned) else None

    if executor is None:
        return _estimate_core_seq(state, moves, slots, pool_cards, consistent,
                                  mode, planned, solve_budget, rng, deadline,
                                  _time)
    try:
        return _estimate_core_par(executor, state, moves, slots, pool_cards,
                                  consistent, mode, planned, solve_budget, rng,
                                  deadline, _time)
    except Exception as exc:
        # Degrade to the in-process path on any pool/worker failure so the
        # request still returns a clean answer.
        from .mclog import log as _log
        _log(f"estimate PARALLEL FAILED, falling back to sequential: "
             f"{type(exc).__name__}: {exc}")
        return _estimate_core_seq(state, moves, slots, pool_cards, consistent,
                                  mode, planned, solve_budget, rng, deadline,
                                  _time)


def _racing_eliminated(wins, evaluated, contenders, enabled=True):
    """Return the set of contender indices that can be eliminated: a move whose
    Wilson UPPER bound is below the current leader's Wilson LOWER bound can't
    catch up. Needs enough evidence (a few evaluated worlds) to be meaningful.
    Never eliminates the current leader, and never drops to zero contenders.

    `enabled` is False for exhaustive/exact runs: there, every arrangement must
    be evaluated for the result to stay exact, so racing is suppressed.
    """
    if not enabled or evaluated < 8 or len(contenders) <= 1:
        return set()

    bounds = {}
    for mi in contenders:
        p, _, half = _wilson_interval(wins[mi], evaluated)
        # Reconstruct the clamped centre so lo/hi match _wilson_interval.
        z2 = 1.96 * 1.96
        denom = 1 + z2 / evaluated
        pc = min(1.0, max(0.0, wins[mi] / evaluated))
        centre = (pc + z2 / (2 * evaluated)) / denom
        lo = max(0.0, centre - half)
        hi = min(1.0, centre + half)
        bounds[mi] = (lo, hi)

    leader = max(contenders, key=lambda mi: wins[mi])
    leader_lo = bounds[leader][0]
    drop = set()
    for mi in contenders:
        if mi == leader:
            continue
        if bounds[mi][1] < leader_lo:
            drop.add(mi)
    # Keep at least the leader as a contender.
    if len(drop) >= len(contenders):
        drop.discard(leader)
    return drop


def _estimate_core_seq(state, moves, slots, pool_cards, consistent, mode,
                       planned, solve_budget, rng, deadline, _time):
    import itertools as _it

    if mode == "exact" or not (slots and consistent):
        worlds_iter = iter([state.clone()])
    elif mode == "exhaustive":
        worlds_iter = (_build_world(state, slots, perm)
                       for perm in _it.permutations(pool_cards))
    else:
        worlds_iter = (_determinize(state, slots, pool_cards, rng)
                       for _ in range(planned))

    wins = [0] * len(moves)
    world_any_win_count = 0
    evaluated = 0
    timed_out = False
    contenders = set(range(len(moves)))
    eliminated: set[int] = set()

    for world in worlds_iter:
        if evaluated >= planned:
            break
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        any_win = False
        aborted = False
        world_wins = []  # indices incremented for THIS world, so we can roll back
        for mi in sorted(contenders):
            mv = moves[mi]
            if deadline is not None and _time.monotonic() >= deadline:
                timed_out = True
                aborted = True
                break
            after = apply_move(world, mv)
            if solve(after, max_states=solve_budget) is not None:
                wins[mi] += 1
                world_wins.append(mi)
                any_win = True
        if aborted:
            # Roll back this world's partial increments so wins stays <=
            # evaluated for every move (otherwise p>1 and the CI math breaks).
            for mi in world_wins:
                wins[mi] -= 1
            break
        if any_win:
            world_any_win_count += 1
        evaluated += 1

        # Racing: retire contenders that provably trail the leader. Suppressed
        # in exhaustive/exact mode so the enumeration stays complete (exact).
        drop = _racing_eliminated(wins, evaluated, contenders,
                                  enabled=(mode == "sampled"))
        if drop:
            eliminated |= drop
            contenders -= drop
        if len(contenders) <= 1 and mode == "sampled":
            break

    return {"wins": wins, "evaluated": evaluated,
            "world_any_win_count": world_any_win_count,
            "timed_out": timed_out, "eliminated": eliminated}


def _estimate_core_par(executor, state, moves, slots, pool_cards, consistent,
                       mode, planned, solve_budget, rng, deadline, _time):
    wins = [0] * len(moves)
    world_any_win_count = 0
    evaluated = 0
    timed_out = False
    contenders = set(range(len(moves)))
    eliminated: set[int] = set()

    batch = max(pool_size() * _BATCH_MULTIPLIER, 1)
    stream = _world_stream(state, slots, pool_cards, consistent, mode, rng)

    while evaluated < planned and len(contenders) > 1:
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        want = min(batch, planned - evaluated)
        worlds = []
        for _ in range(want):
            try:
                worlds.append(next(stream))
            except StopIteration:
                break
        if not worlds:
            break

        # Only test the current contenders in the workers — that's where racing
        # saves real work. Every contender is tested on the SAME batch of
        # worlds, preserving fair comparison among them.
        move_tuple = tuple(moves[mi] for mi in sorted(contenders))
        idx_order = sorted(contenders)
        futures = _submit_batch(executor, worlds, eval_world_moves,
                                (move_tuple, solve_budget))

        # Each future is one world evaluated ATOMICALLY across all contenders,
        # so a completed future advances every contender together (fair), and
        # dropping an unfinished one at the deadline drops that whole world —
        # keeping wins <= evaluated.
        for res, cut in _drain_batch(futures, deadline, _time):
            if cut:
                timed_out = True
                break
            any_win = False
            for slot, mi in enumerate(idx_order):
                if res[slot]:
                    wins[mi] += 1
                    any_win = True
            if any_win:
                world_any_win_count += 1
            evaluated += 1
        if timed_out:
            break

        drop = _racing_eliminated(wins, evaluated, contenders,
                                  enabled=(mode == "sampled"))
        if drop:
            eliminated |= drop
            contenders -= drop

    return {"wins": wins, "evaluated": evaluated,
            "world_any_win_count": world_any_win_count,
            "timed_out": timed_out, "eliminated": eliminated}


def mc_plan_steps(state: GameState, n: int = 3, samples: int = 60,
                  solve_budget: int = 60_000, rng: random.Random | None = None,
                  time_budget_s: float = 8.0) -> dict:
    """Monte Carlo analog of strategy.plan_steps: recommend up to n steps where
    the FIRST move is the MC-best move, and the rest follow the exact solver's
    winning line within the most-favorable sampled world (for continuity).

    Shape matches strategy.plan_steps plus MC fields (winProbability, perMove).
    """
    from .strategy import flips_hidden_card

    est = estimate(state, samples=samples, solve_budget=solve_budget, rng=rng,
                   time_budget_s=time_budget_s)

    if est.get("won"):
        return {"solved": True, "gameOver": False, "stuck": False, "steps": [],
                "revealNext": False, "winProbability": 1.0, "perMove": [],
                "mc": True}
    if est["gameOver"]:
        return {"solved": False, "gameOver": True,
                "stuck": est.get("stuck", False), "steps": [],
                "revealNext": False, "winProbability": 0.0, "perMove": [],
                "mc": True}

    best = est["best"]
    if best is None:
        return {"solved": False, "gameOver": True, "stuck": False, "steps": [],
                "revealNext": False, "winProbability": 0.0,
                "perMove": est["perMove"], "mc": True}

    best_move = Move(**best)
    reveal_next = flips_hidden_card(state, best_move)

    steps = [{
        "index": 0,
        "kind": best_move.kind,
        "description": best_move.describe(),
        "revealsHidden": reveal_next,
        "move": best,
    }]

    # For steps 2..n, continue the line in a representative winnable world using
    # the exact solver, so the user sees a plausible short continuation. We stop
    # at a reveal (unknown flip) since beyond that depends on the real card.
    if not reveal_next and len(est["perMove"]) and est["perMove"][0]["winRate"] > 0:
        slots = _hidden_slots(state)
        pool = _unknown_pool(state)
        rng2 = rng or random.Random()
        # find a world where best_move leads to a win, then read its line
        for _ in range(samples):
            world = _determinize(state, slots, pool, rng2) if (slots and len(pool) == len(slots)) else state.clone()
            after = apply_move(world, best_move)
            line = solve(after, max_states=solve_budget)
            if line is not None:
                sim = after
                for mv in line[: n - 1]:
                    rv = flips_hidden_card(sim, mv)
                    steps.append({
                        "index": len(steps),
                        "kind": mv.kind,
                        "description": mv.describe(),
                        "revealsHidden": rv,
                        "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                                 "count": mv.count, "card": mv.card},
                    })
                    if rv:
                        break
                    sim = apply_move(sim, mv)
                break

    return {
        "solved": False, "gameOver": False, "stuck": False,
        "steps": steps[:n],
        "revealNext": reveal_next,
        "winProbability": est["winProbability"],
        "winMargin": est.get("winMargin", 0.0),
        "exact": est.get("exact", False),
        "timedOut": est.get("timedOut", False),
        "samples": est.get("samples", 0),
        "perMove": est["perMove"],
        "mc": True,
    }
