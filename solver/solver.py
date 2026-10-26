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
    """Return a list of moves that wins, or None if not found within budget.

    Weighted best-first search with transposition pruning. Instead of carrying
    the whole move list in every heap entry (an O(path) copy per push), each
    node stores only its depth and a reference into a came-from map
    (``prev[key] = (parent_key, move)``); the winning line is reconstructed by
    walking that map once, only when a win is found. A closed set guarantees a
    popped state is expanded at most once. All structures are LOCAL to this call
    (no shared/global caches), so parallel/pickled worlds never interfere and
    the search stays deterministic for a given start state.
    """
    if start.is_won():
        return []

    counter = itertools.count()
    start_key = start.key()
    # heap entries: (priority, tiebreak, depth, state, state_key)
    heap: list[tuple[int, int, int, GameState, tuple]] = []
    heapq.heappush(heap, (_heuristic(start), next(counter), 0, start, start_key))
    # best_seen: shortest known depth to reach a key (transposition table).
    best_seen: dict[tuple, int] = {start_key: 0}
    # came-from map for path reconstruction on a win (parent pointers).
    prev: dict[tuple, tuple] = {start_key: (None, None)}
    # closed set: keys already popped/expanded, never re-expanded.
    closed: set[tuple] = set()
    explored = 0

    while heap and explored < max_states:
        _, _, depth, state, state_key = heapq.heappop(heap)

        # Skip stale heap entries (a better path to this key was found after
        # this one was pushed) and anything already expanded.
        if state_key in closed:
            continue
        closed.add(state_key)
        explored += 1

        if state.is_won():
            return _reconstruct(prev, state_key)

        if depth > 400:  # avoid runaway lines
            continue

        new_len = depth + 1
        for move in legal_moves(state):
            nxt = apply_move(state, move)
            key = nxt.key()
            # Prune duplicates before any heap work: skip if already expanded,
            # or reachable at least as cheaply by a path we've already queued.
            if key in closed:
                continue
            seen = best_seen.get(key)
            if seen is not None and seen <= new_len:
                continue
            best_seen[key] = new_len
            prev[key] = (state_key, move)
            priority = _heuristic(nxt) + _move_priority(move) + new_len // 8
            heapq.heappush(heap, (priority, next(counter), new_len, nxt, key))

    return None


def is_winnable(start: GameState, max_states: int = 200_000) -> bool:
    """Boolean fast path: is `start` winnable within the state budget?

    Same search as `solve` but without any path bookkeeping — no came-from map
    and no reconstruction — since callers that only need yes/no (the Monte
    Carlo `position_estimate` / `eval_world_winnable` worker) never use the move
    list. Returns the SAME verdict as ``solve(...) is not None`` for a given
    state (identical exploration order and pruning); it just skips building the
    winning line. All structures are local to the call, so it is deterministic
    and safe across parallel/pickled worlds.
    """
    if start.is_won():
        return True

    counter = itertools.count()
    start_key = start.key()
    heap: list[tuple[int, int, int, GameState, tuple]] = []
    heapq.heappush(heap, (_heuristic(start), next(counter), 0, start, start_key))
    best_seen: dict[tuple, int] = {start_key: 0}
    closed: set[tuple] = set()
    explored = 0

    while heap and explored < max_states:
        _, _, depth, state, state_key = heapq.heappop(heap)
        if state_key in closed:
            continue
        closed.add(state_key)
        explored += 1

        if state.is_won():
            return True
        if depth > 400:
            continue

        new_len = depth + 1
        for move in legal_moves(state):
            nxt = apply_move(state, move)
            key = nxt.key()
            if key in closed:
                continue
            seen = best_seen.get(key)
            if seen is not None and seen <= new_len:
                continue
            best_seen[key] = new_len
            priority = _heuristic(nxt) + _move_priority(move) + new_len // 8
            heapq.heappush(heap, (priority, next(counter), new_len, nxt, key))

    return False


def _reconstruct(prev: dict[tuple, tuple], goal_key: tuple) -> list[Move]:
    """Walk the came-from map from the winning state back to the start and
    return the moves in forward order."""
    moves: list[Move] = []
    key = goal_key
    while True:
        parent_key, move = prev[key]
        if move is None:
            break
        moves.append(move)
        key = parent_key
    moves.reverse()
    return moves


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
