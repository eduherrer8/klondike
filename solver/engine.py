"""Move generation and application for Klondike."""

from __future__ import annotations

from .cards import Card, can_stack_foundation, can_stack_tableau, is_unknown
from .game import GameState, Move


def _flip_if_needed(col: list[tuple[Card, bool]]) -> None:
    """After removing cards, ensure the new top card is face-up."""
    if col and not col[-1][1]:
        card, _ = col[-1]
        col[-1] = (card, True)


def apply_move(state: GameState, move: Move) -> GameState:
    """Return a new state with the move applied. Does not validate heavily;
    legal_moves is the source of truth for legality."""
    s = state.clone()

    if move.kind == "draw":
        if not s.stock:
            # recycle waste back into stock (preserving order)
            s.stock = s.waste[::-1]
            s.waste = []
            s.fan_size = 0
        else:
            n = min(s.draw_count, len(s.stock))
            drawn = [s.stock.pop() for _ in range(n)]  # pop from end
            s.waste.extend(drawn)
            s.fan_size = n  # the current visible fan is these n cards
        return s

    if move.kind == "tableau":
        src_col = s.tableau[move.src]
        moving = src_col[-move.count:]
        s.tableau[move.src] = src_col[:-move.count]
        _flip_if_needed(s.tableau[move.src])
        s.tableau[move.dst].extend(moving)
        return s

    if move.kind == "to_foundation":
        src_col = s.tableau[move.src]
        card, _ = src_col.pop()
        s.tableau[move.src] = src_col
        _flip_if_needed(s.tableau[move.src])
        s.foundations[card.suit].append(card)
        return s

    if move.kind == "waste_to_tableau":
        card = s.waste.pop()
        s.tableau[move.dst].append((card, True))
        s.fan_size = max(0, s.fan_size - 1)  # peeled one off the current fan
        return s

    if move.kind == "waste_to_foundation":
        card = s.waste.pop()
        s.foundations[card.suit].append(card)
        s.fan_size = max(0, s.fan_size - 1)
        return s

    if move.kind == "foundation_to_tableau":
        # src holds the suit index here via card code
        card = Card.parse(move.card)
        s.foundations[card.suit].pop()
        s.tableau[move.dst].append((card, True))
        return s

    raise ValueError(f"unknown move kind: {move.kind}")


def legal_moves(state: GameState) -> list[Move]:
    moves: list[Move] = []

    # 1. Tableau card(s) -> foundation (only the single top card)
    for col_i, col in enumerate(state.tableau):
        if not col:
            continue
        card, up = col[-1]
        if up and not is_unknown(card) and \
                can_stack_foundation(card, state.foundation_top(card.suit), card.suit):
            moves.append(Move("to_foundation", src=col_i, card=card.code()))

    # 2. Waste top -> foundation
    if state.waste:
        w = state.waste[-1]
        if not is_unknown(w) and \
                can_stack_foundation(w, state.foundation_top(w.suit), w.suit):
            moves.append(Move("waste_to_foundation", card=w.code()))

    # 3. Tableau run -> another tableau.
    # This variant does NOT allow splitting a column's face-up run: a
    # tableau-to-tableau move always moves the ENTIRE face-up run of the
    # source column. The bottom of that run is run[0].
    for src_i, col in enumerate(state.tableau):
        run = state.column_face_up_run(src_i)
        if not run:
            continue
        bottom = run[0]
        count = len(run)
        has_facedown_below = count < len(col)  # face-down cards remain beneath the run
        for dst_i, dst_col in enumerate(state.tableau):
            if dst_i == src_i:
                continue
            if not dst_col:
                # Only a King may move to an empty column. Moving a King-led
                # run from a column that has nothing beneath it (the run is the
                # whole column) just relocates it to another empty column, a
                # pointless loop, so skip that case.
                if bottom.rank == 13 and has_facedown_below:
                    moves.append(Move("tableau", src=src_i, dst=dst_i, count=count, card=bottom.code()))
            else:
                top_card, top_up = dst_col[-1]
                if top_up and not is_unknown(top_card) and can_stack_tableau(bottom, top_card):
                    moves.append(Move("tableau", src=src_i, dst=dst_i, count=count, card=bottom.code()))

    # 4. Waste top -> tableau
    if state.waste:
        w = state.waste[-1]
        for dst_i, dst_col in enumerate(state.tableau):
            if not dst_col:
                if w.rank == 13:
                    moves.append(Move("waste_to_tableau", dst=dst_i, card=w.code()))
            else:
                top_card, top_up = dst_col[-1]
                if top_up and not is_unknown(top_card) and can_stack_tableau(w, top_card):
                    moves.append(Move("waste_to_tableau", dst=dst_i, card=w.code()))

    # 5. Draw (or recycle) from stock
    if state.stock or state.waste:
        moves.append(Move("draw"))

    return moves
