"""A best-first search solver for Klondike solitaire.

The solver uses a weighted depth-limited best-first search with state dedupe.
It is not guaranteed to find the optimal (shortest) solution, but it finds a
winning line for most solvable deals, and returns the sequence of moves so the
UI can step through them.
"""

from __future__ import annotations

import heapq
import itertools

from .engine import apply_move, legal_moves
from .game import GameState, Move


def _heuristic(state: GameState) -> int:
    """Lower is better. Reward cards on foundations; penalize buried cards."""
    foundation_cards = sum(len(p) for p in state.foundations.values())
    face_down = sum(1 for col in state.tableau for _, up in col if not up)
    stock_waste = len(state.stock) + len(state.waste)
    # 52 foundation cards is the goal.
    return (52 - foundation_cards) * 3 + face_down * 2 + stock_waste


def _move_priority(move: Move) -> int:
    """Secondary ordering so productive moves are explored first."""
    order = {
        "waste_to_foundation": 0,
        "to_foundation": 0,
        "tableau": 1,
        "waste_to_tableau": 2,
        "foundation_to_tableau": 4,
        "draw": 3,
    }
    return order.get(move.kind, 5)


def solve(start: GameState, max_states: int = 200_000) -> list[Move] | None:
    """Return a list of moves that wins, or None if not found within budget."""
    if start.is_won():
        return []

    counter = itertools.count()
    start_key = start.key()
    # heap entries: (heuristic, tiebreak, state, path)
    heap: list[tuple[int, int, GameState, list[Move]]] = []
    heapq.heappush(heap, (_heuristic(start), next(counter), start, []))
    best_seen: dict[tuple, int] = {start_key: 0}
    explored = 0

    while heap and explored < max_states:
        h, _, state, path = heapq.heappop(heap)
        explored += 1

        if state.is_won():
            return path

        if len(path) > 400:  # avoid runaway lines
            continue

        for move in legal_moves(state):
            nxt = apply_move(state, move)
            key = nxt.key()
            new_len = len(path) + 1
            prev = best_seen.get(key)
            if prev is not None and prev <= new_len:
                continue
            best_seen[key] = new_len
            priority = _heuristic(nxt) + _move_priority(move) + new_len // 8
            heapq.heappush(heap, (priority, next(counter), nxt, path + [move]))

    return None


def _steps_payload(solution: list[Move], n: int, total: int) -> dict:
    steps = []
    for i, move in enumerate(solution[:n]):
        steps.append({
            "index": i,
            "kind": move.kind,
            "description": move.describe(),
            "move": {
                "kind": move.kind,
                "src": move.src,
                "dst": move.dst,
                "count": move.count,
                "card": move.card,
            },
        })
    return {"solved": True, "total_moves": total, "steps": steps}


def next_steps(start: GameState, n: int = 3, max_states: int = 200_000) -> dict:
    """Compute a full solution and return the first n steps for the UI.

    Returns a dict with keys:
      - solved: bool
      - total_moves: int (full solution length, if solved)
      - steps: list of {index, kind, description, move}
    """
    solution = solve(start, max_states=max_states)
    if solution is None:
        return {"solved": False, "total_moves": 0, "steps": []}
    return _steps_payload(solution, n, len(solution))


def solve_midgame(layout, max_states: int = 200_000, max_arrangements: int = 400):
    """Solve a mid-game position that may contain face-down unknown cards.

    Tries arrangements of the unknown pool into the face-down slots and returns
    the first winning line found, along with the assignment that achieves it.

    Returns (solution, assignment) where solution is a list[Move] and
    assignment maps (col, pos) -> Card, or (None, None) if no arrangement in the
    tried budget is solvable.
    """
    import itertools as _it

    from .resolver import build_state

    slots = layout.unknown_slots
    pool = layout.unknown_pool

    # No hidden cards: a single deterministic solve.
    if not slots:
        state = build_state(layout, {})
        sol = solve(state, max_states=max_states)
        return (sol, {}) if sol is not None else (None, None)

    tried = 0
    for perm in _it.permutations(pool):
        if tried >= max_arrangements:
            break
        tried += 1
        assignment = {slot: card for slot, card in zip(slots, perm)}
        state = build_state(layout, assignment)
        sol = solve(state, max_states=max_states)
        if sol is not None:
            return sol, assignment

    return None, None


def midgame_steps(layout, n: int = 3, max_states: int = 200_000,
                  max_arrangements: int = 400) -> dict:
    """Solve a mid-game layout and return the first n steps for the UI.

    On success also returns the `assignment` (hidden-card guesses) that make the
    position winnable, so the caller can seed the game state for stepping.
    """
    solution, assignment = solve_midgame(
        layout, max_states=max_states, max_arrangements=max_arrangements
    )
    if solution is None:
        return {"solved": False, "total_moves": 0, "steps": [], "assignment": None}
    payload = _steps_payload(solution, n, len(solution))
    payload["assignment"] = {
        f"{col},{pos}": card.code() for (col, pos), card in assignment.items()
    }
    return payload
