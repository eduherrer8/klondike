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
from .solver import solve


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
    if num_hidden <= 14:
        return {"mode": "sampled", "samples": max(40, min(requested_samples, 80))}
    return {"mode": "sampled", "samples": min(requested_samples, 60)}


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


def position_estimate(state: GameState, samples: int = 60,
                      solve_budget: int = 60_000,
                      rng: random.Random | None = None,
                      time_budget_s: float = 8.0) -> dict:
    """CHEAP win-probability for the CURRENT position.

    Unlike `estimate`, this does NOT evaluate every legal move. It samples the
    position once per world and asks "is this world winnable at all?" — one
    solve per world instead of (moves x worlds). Use this for the
    win-probability readout; use `estimate` only when you need to compare moves.
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

    winnable = 0
    evaluated = 0
    timed_out = False
    for world in _world_stream(state, slots, pool, consistent, mode, rng):
        if evaluated >= planned:
            break
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        if solve(world, max_states=solve_budget) is not None:
            winnable += 1
        evaluated += 1

    exact = (mode in ("exhaustive", "exact")) and not timed_out
    p, margin, _ = _wilson_interval(winnable, evaluated)
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

    import itertools as _it

    slots = _hidden_slots(state)
    pool = _unknown_pool(state)
    consistent = (len(pool) == len(slots))
    num_hidden = len(slots) if consistent else 0

    # Adaptive effort: enumerate exactly when the arrangement space is tiny,
    # otherwise sample in the accepted ~40-80 range (capped for heavy cases).
    effort = _choose_effort(num_hidden, samples)
    mode = effort["mode"]
    planned = effort["samples"]

    # Build the stream of worlds to evaluate.
    if num_hidden == 0 or not consistent:
        worlds_iter = iter([state.clone()])
        planned = 1
        mode = "exact"  # fully known position: a single deterministic world
    elif mode == "exhaustive":
        worlds_iter = (_build_world(state, slots, perm)
                       for perm in _it.permutations(pool))
    else:
        worlds_iter = (_determinize(state, slots, pool, rng) for _ in range(planned))

    # Evaluate world-by-world so the wall-clock cap can stop early with every
    # move having been tried on the SAME number of worlds (fair comparison).
    wins = [0] * len(moves)
    world_any_win_count = 0
    evaluated = 0
    timed_out = False

    for world in worlds_iter:
        if evaluated >= planned:
            break
        if deadline is not None and _time.monotonic() >= deadline:
            timed_out = True
            break
        any_win = False
        aborted = False
        world_wins = []  # indices incremented for THIS world, so we can roll back
        for mi, mv in enumerate(moves):
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
        })

    per_move.sort(key=lambda m: m["winRate"], reverse=True)
    win_probability = (world_any_win_count / evaluated) if evaluated else 0.0
    _, wp_margin, _ = _wilson_interval(world_any_win_count, evaluated)
    best = per_move[0]["move"] if per_move else None

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
        "perMove": est["perMove"],
        "mc": True,
    }
