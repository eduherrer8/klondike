"""Strategy-based planner for mid-game positions with hidden cards.

Unlike the exact `solve()` (which needs every card known), this planner treats
face-down cards as OPAQUE unknowns. It never guesses their identity. Instead it
reasons only about what is actually known — the visible tableau cards and the
full (known) stock/waste cycling sequence — and recommends the strongest moves
by a strategy heuristic, in the spirit of good Klondike play:

  * Prefer moves that EXPOSE a face-down card (the only way to gain info).
  * Prefer creating/using empty columns productively (Kings only).
  * Send cards to the foundation ONLY when "safe" (conservative autoplay,
    option a): a card is safe to play up when it can no longer be needed to
    park an opposite-color card on the tableau.
  * Use the known stock: a draw that brings a useful card to the waste top is
    valued; aimless draws are cheap but allowed so the line can progress.
  * Avoid moves with no purpose (shuffling runs that reveal nothing).

A move that flips a face-down card is a REVEAL point: after it, the planner
stops the line (the flipped card is unknown) and the move is recommended so the
user can reveal the real card and re-plan.

"Game over" means there are literally no legal moves at all.
"""

from __future__ import annotations

import heapq
import itertools

from .cards import Card, is_unknown
from .engine import apply_move, legal_moves
from .game import GameState, Move

RANK_K = 13


# ---- safe autoplay (conservative foundation rule) ----

def _min_foundation_rank_opposite(state: GameState, red: bool) -> int:
    """Lowest foundation top rank among the two OPPOSITE-color suits.

    Standard safe-autoplay uses the opposite-color foundations to decide if a
    card can still be needed on the tableau. A card of rank r is safe to send
    up if both opposite-color foundations are already at >= r - 1 (so nothing
    of rank r-1 needs this card as a parking spot)."""
    if red:
        suits = ("S", "C")  # opposite of red is black
    else:
        suits = ("H", "D")  # opposite of black is red
    return min(len(state.foundations[s]) for s in suits)


def is_safe_to_foundation(state: GameState, card: Card) -> bool:
    """Conservative rule: Aces and twos are always safe; otherwise the card is
    safe only if both opposite-color foundations have reached at least rank-1."""
    if card.rank <= 2:
        return True
    return _min_foundation_rank_opposite(state, card.is_red) >= card.rank - 1


# ---- move classification & reveal detection ----

def flips_hidden_card(state: GameState, move: Move) -> bool:
    """Would applying `move` expose a face-down (unknown) card as the new top
    of a column?"""
    if move.kind in ("tableau", "to_foundation"):
        col = state.tableau[move.src]
        if move.kind == "to_foundation":
            removed = 1
        else:
            removed = move.count
        remaining = len(col) - removed
        if remaining <= 0:
            return False  # column becomes empty, nothing to flip
        # the card that becomes the new top:
        card, up = col[remaining - 1]
        return (not up) and is_unknown(card)
    return False


# ---- strategy scoring ----

def _foundation_count(state: GameState) -> int:
    return sum(len(p) for p in state.foundations.values())


def _hidden_count(state: GameState) -> int:
    return sum(1 for col in state.tableau for c, up in col if (not up) and is_unknown(c))


def _empty_columns(state: GameState) -> int:
    return sum(1 for col in state.tableau if not col)


def move_score(state: GameState, move: Move) -> float:
    """Immediate desirability of a move (higher is better). Used to rank the
    first move and to guide the search."""
    score = 0.0

    if move.kind in ("to_foundation", "waste_to_foundation"):
        card = _move_card(state, move)
        # Conservative (option a): strongly prefer SAFE foundation plays; only
        # mildly discourage unsafe ones. The discouragement must stay ABOVE a
        # draw's score, so an unsafe foundation play is still chosen over
        # looping the stock when it's the only way to make progress.
        if card is not None and is_safe_to_foundation(state, card):
            score += 6.0
        else:
            score += 0.5  # mild: better than a dead draw, worse than safe plays

    if move.kind in ("tableau", "to_foundation") and flips_hidden_card(state, move):
        score += 12.0  # exposing a hidden card is the highest-value action

    if move.kind == "tableau":
        # moving a run that empties its source column (exposing nothing because
        # there was nothing below) is only good if it frees an empty column by
        # relocating a King off face-down cards; otherwise mild.
        src = state.tableau[move.src]
        if len(src) == move.count:
            score += 1.0  # creates an empty column
        else:
            score += 2.0  # likely uncovering progress

    if move.kind == "waste_to_tableau":
        score += 3.0  # getting a card out of the waste onto the tableau is useful

    if move.kind == "foundation_to_tableau":
        score -= 2.0  # only do this when it clearly unblocks; generally avoid

    if move.kind == "draw":
        score -= 0.1  # cheap, allowed, but not inherently productive

    return score


def _move_card(state: GameState, move: Move) -> Card | None:
    if move.kind == "to_foundation":
        c, _ = state.tableau[move.src][-1]
        return None if is_unknown(c) else c
    if move.kind == "waste_to_foundation":
        return state.waste[-1] if state.waste else None
    if move.card:
        try:
            return Card.parse(move.card)
        except ValueError:
            return None
    return None


def state_score(state: GameState) -> float:
    """Heuristic value of a position (higher is better)."""
    return (
        _foundation_count(state) * 5.0      # progress toward the goal
        - _hidden_count(state) * 3.0         # fewer hidden cards is better
        + _empty_columns(state) * 2.0        # flexibility
        - len(state.waste) * 0.1             # prefer not piling up the waste
    )


# ---- planner ----

def _lookahead_value(state: GameState, depth: int, max_depth: int,
                     seen: set) -> float:
    """Shallow greedy lookahead: value of a position = its own score plus the
    best achievable score within `max_depth` known moves (reveal moves are
    treated as terminal with a bonus). Used to break ties between first moves
    and to steer away from dead ends."""
    base = state_score(state)
    if depth >= max_depth:
        return base
    best = base
    for mv in legal_moves(state):
        if flips_hidden_card(state, mv):
            best = max(best, base + 12.0)  # reveal is valuable, terminal here
            continue
        if mv.kind == "draw":
            # don't recurse deeply through pure draws; value them shallowly
            nxt = apply_move(state, mv)
            best = max(best, state_score(nxt) - 0.2)
            continue
        nxt = apply_move(state, mv)
        k = nxt.key()
        if k in seen:
            continue
        seen = seen | {k}
        best = max(best, _lookahead_value(nxt, depth + 1, max_depth, seen) - 0.1)
    return best


def _rank_moves(state: GameState, lookahead: int) -> list[tuple[float, Move]]:
    """Rank legal moves by immediate strategy score, with a shallow lookahead
    value as a tiebreaker/dead-end guard."""
    ranked: list[tuple[float, Move]] = []
    for mv in legal_moves(state):
        immediate = move_score(state, mv)
        if flips_hidden_card(state, mv):
            future = state_score(state) + 12.0  # reveal: value the exposure
        else:
            nxt = apply_move(state, mv)
            future = _lookahead_value(nxt, 0, lookahead, {nxt.key()})
        # Weight immediate strategy highly; use future to break ties / avoid
        # moves that lead nowhere.
        ranked.append((immediate * 10.0 + future, mv))
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked


def plan(state: GameState, max_depth: int = 8, max_nodes: int = 30000) -> list[Move]:
    """Return a recommended sequence of moves (a best line) from `state`.

    Greedy by immediate strategy value with a shallow lookahead tiebreaker. The
    line stops at the first move that flips a hidden card (a reveal point) or
    after a few moves, since after a reveal the plan is recomputed anyway. The
    returned list is empty only when there are no legal moves (game over).
    """
    line: list[Move] = []
    sim = state
    visited: set = {state.key()}
    for _ in range(max_depth):
        moves = legal_moves(sim)
        if not moves:
            break
        ranked = _rank_moves(sim, lookahead=3)
        # pick the best move that doesn't immediately loop back to a seen state
        chosen = None
        for _val, mv in ranked:
            if flips_hidden_card(sim, mv):
                chosen = mv
                break
            nxt = apply_move(sim, mv)
            if nxt.key() in visited:
                continue
            chosen = mv
            break
        if chosen is None:
            chosen = ranked[0][1]  # all lead to seen states; take the top one

        line.append(chosen)
        if flips_hidden_card(sim, chosen):
            break  # reveal point: stop, user will reveal and we re-plan
        sim = apply_move(sim, chosen)
        if sim.key() in visited:
            break  # avoid cycling
        visited.add(sim.key())

    return line


def is_dead_stock_cycle(state: GameState) -> bool:
    """True if the ONLY thing you can do is cycle the stock/waste and that
    cycling never exposes a card that enables any non-draw move.

    This catches the real "stuck" case: every tableau top is frozen and the
    stock just loops the same unplayable cards. It is a dead end even though a
    'draw' is technically legal.
    """
    moves = legal_moves(state)
    if not moves:
        return False  # truly no moves -> handled as game over elsewhere
    if any(m.kind != "draw" for m in moves):
        return False  # a real move is available right now; not a dead cycle

    # Only draws are available right now. Explore the reachable space using
    # draws AND the waste-peel plays they enable (standard draw-3: playing the
    # waste top exposes the next card behind it). If ANY sequence of draws and
    # waste plays ever leads to a state with a PRODUCTIVE move (a tableau move,
    # or any foundation move), the position is NOT a dead cycle.
    #
    # A "productive" move is any legal move that is not a draw and not a pure
    # waste shuffle. Reaching one means the stock can still unlock the game.
    start = state
    seen: set = {_stock_key(start)}
    frontier = [start]
    guard = 0
    MAX = 20000
    while frontier and guard < MAX:
        cur = frontier.pop()
        guard += 1
        for mv in legal_moves(cur):
            if mv.kind in ("to_foundation", "waste_to_foundation",
                           "tableau", "waste_to_tableau",
                           "foundation_to_tableau"):
                # Playing a waste card onto the tableau/foundation, or any
                # tableau/foundation move becoming available, is real progress.
                return False
            if mv.kind == "draw":
                nxt = apply_move(cur, mv)
                k = _stock_key(nxt)
                if k not in seen:
                    seen.add(k)
                    frontier.append(nxt)
    # Explored every reachable draw-state and never found a productive move.
    return True


def _stock_key(state: GameState) -> tuple:
    """Dedupe key capturing only the stock/waste (tableau/foundations are fixed
    while we're exploring a stuck-cycle: no productive move has happened)."""
    return (
        tuple((c.rank, c.suit) for c in state.stock),
        tuple((c.rank, c.suit) for c in state.waste),
    )


def plan_steps(state: GameState, n: int = 3, max_depth: int = 12,
               max_nodes: int = 30000) -> dict:
    """Plan and return the first n recommended steps for the UI.

    Returns:
      - solved: True if the position is already won
      - gameOver: True if there are no legal moves, OR the only option is a
                  dead stock cycle that never unlocks anything
      - stuck: True specifically for the dead-stock-cycle case (vs. no moves)
      - steps: up to n recommended moves
      - revealNext: True if the next step flips a hidden card (UI should prompt)
    """
    if state.is_won():
        return {"solved": True, "gameOver": False, "stuck": False,
                "steps": [], "revealNext": False}

    if not legal_moves(state):
        return {"solved": False, "gameOver": True, "stuck": False,
                "steps": [], "revealNext": False}

    if is_dead_stock_cycle(state):
        return {"solved": False, "gameOver": True, "stuck": True,
                "steps": [], "revealNext": False}

    line = plan(state, max_depth=max_depth, max_nodes=max_nodes)
    steps = []
    sim = state
    reveal_next = False
    for i, mv in enumerate(line[:n]):
        reveal = flips_hidden_card(sim, mv)
        steps.append({
            "index": i,
            "kind": mv.kind,
            "description": mv.describe(),
            "revealsHidden": reveal,
            "move": {"kind": mv.kind, "src": mv.src, "dst": mv.dst,
                     "count": mv.count, "card": mv.card},
        })
        if i == 0:
            reveal_next = reveal
        if reveal:
            break  # can't simulate past an unknown flip
        sim = apply_move(sim, mv)

    return {"solved": False, "gameOver": False, "stuck": False,
            "steps": steps, "revealNext": reveal_next}
