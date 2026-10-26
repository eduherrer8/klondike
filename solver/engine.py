"""Move generation and application for Klondike."""

from __future__ import annotations

from .cards import UNKNOWN, Card, can_stack_foundation, can_stack_tableau, is_unknown
from .game import GameState, Move


def _flip_top(col: list[tuple[Card, bool]]) -> None:
    """Flip the new top card face-up in place (col is a fresh, owned list)."""
    if col and not col[-1][1]:
        card, _ = col[-1]
        col[-1] = (card, True)


def apply_move(state: GameState, move: Move) -> GameState:
    """Return a NEW state with the move applied, using copy-on-write.

    Rather than deep-cloning the whole position for every candidate move (the
    old hot-path cost), this shares the parent's unchanged structures by
    reference and copies only the tableau columns / foundation pile the move
    actually touches. Correctness rests on an invariant the solver upholds:
    GameStates are treated as immutable after construction, so sharing a list
    object between parent and child is safe as long as we never mutate a shared
    list — every branch below copies a structure before changing it.

    `legal_moves` is the source of truth for legality; this does not re-validate.
    """
    kind = move.kind
    tab = state.tableau  # shared unless a column changes

    if kind == "draw":
        if not state.stock:
            # recycle waste back into stock (preserving order)
            return GameState(
                tableau=tab, foundations=state.foundations,
                stock=state.waste[::-1], waste=[], draw_count=state.draw_count,
                fan_size=0,
            )
        n = min(state.draw_count, len(state.stock))
        new_stock = state.stock[:-n]
        drawn = state.stock[-1:-n - 1:-1]  # last n, in pop order (end first)
        return GameState(
            tableau=tab, foundations=state.foundations,
            stock=new_stock, waste=state.waste + drawn,
            draw_count=state.draw_count, fan_size=n,
        )

    if kind == "tableau":
        new_tab = list(tab)
        src_col = tab[move.src]
        moving = src_col[-move.count:]
        shortened = src_col[:-move.count]
        _flip_top(shortened)
        new_tab[move.src] = shortened
        new_tab[move.dst] = tab[move.dst] + moving
        return GameState(
            tableau=new_tab, foundations=state.foundations,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size,
        )

    if kind == "to_foundation":
        new_tab = list(tab)
        src_col = list(tab[move.src])
        card, _ = src_col.pop()
        _flip_top(src_col)
        new_tab[move.src] = src_col
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit] + [card]
        return GameState(
            tableau=new_tab, foundations=new_found,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size,
        )

    if kind == "waste_to_tableau":
        card = state.waste[-1]
        new_tab = list(tab)
        new_tab[move.dst] = tab[move.dst] + [(card, True)]
        return GameState(
            tableau=new_tab, foundations=state.foundations,
            stock=state.stock, waste=state.waste[:-1],
            draw_count=state.draw_count, fan_size=max(0, state.fan_size - 1),
        )

    if kind == "waste_to_foundation":
        card = state.waste[-1]
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit] + [card]
        return GameState(
            tableau=tab, foundations=new_found,
            stock=state.stock, waste=state.waste[:-1],
            draw_count=state.draw_count, fan_size=max(0, state.fan_size - 1),
        )

    if kind == "foundation_to_tableau":
        card = Card.parse(move.card)
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit][:-1]
        new_tab = list(tab)
        new_tab[move.dst] = tab[move.dst] + [(card, True)]
        return GameState(
            tableau=new_tab, foundations=new_found,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size,
        )

    raise ValueError(f"unknown move kind: {move.kind}")


def legal_moves(state: GameState) -> list[Move]:
    moves: list[Move] = []
    tableau = state.tableau
    foundations = state.foundations

    # Each destination column's exposed (top_card, playable?) computed once, so
    # the run-placement and waste-placement loops below don't re-inspect tops.
    # playable means: a face-up KNOWN card you may stack onto.
    dst_tops: list[tuple[Card | None, bool]] = []
    for col in tableau:
        if not col:
            dst_tops.append((None, False))
            continue
        tc, tup = col[-1]
        dst_tops.append((tc, tup and tc is not UNKNOWN and tc.kid is not None))

    # 1. Tableau card(s) -> foundation (only the single top card)
    for col_i, col in enumerate(tableau):
        if not col:
            continue
        card, up = col[-1]
        if up and card is not UNKNOWN and card.kid is not None and \
                can_stack_foundation(card, state.foundation_top(card.suit), card.suit):
            moves.append(Move("to_foundation", src=col_i, card=card.code()))

    # 2. Waste top -> foundation
    waste = state.waste
    w = waste[-1] if waste else None
    w_known = w is not None and w is not UNKNOWN and w.kid is not None
    if w_known and can_stack_foundation(w, state.foundation_top(w.suit), w.suit):
        moves.append(Move("waste_to_foundation", card=w.code()))

    # 3. Tableau run -> another tableau.
    # This variant does NOT allow splitting a column's face-up run: a
    # tableau-to-tableau move always moves the ENTIRE face-up run of the source
    # column. The movable run is the maximal block of contiguous face-up KNOWN
    # cards at the top; an UNKNOWN or any face-down card below ends it. We scan
    # each column from the top down once, tracking only the run's bottom card
    # and length (no list is materialized) — this was the hottest spot.
    for src_i in range(7):
        col = tableau[src_i]
        n = len(col)
        if not n:
            continue
        count = 0
        bottom = None
        # Walk from the top toward the bottom collecting face-up known cards;
        # stop at the first face-down or opaque-unknown card. The collected
        # block is exactly column_face_up_run's trailing run.
        for j in range(n - 1, -1, -1):
            card, up = col[j]
            if not up or card is UNKNOWN or card.kid is None:
                break
            bottom = card
            count += 1
        if count == 0:
            # Top is face-down or an opaque unknown: nothing movable here.
            continue
        has_facedown_below = count < n
        bottom_is_king = bottom.rank == 13
        bottom_code = bottom.code()
        for dst_i in range(7):
            if dst_i == src_i:
                continue
            top_card, playable = dst_tops[dst_i]
            if top_card is None:
                # Only a King may move to an empty column. Moving a King-led run
                # from a column with nothing beneath it just relocates it, a
                # pointless loop, so require face-down cards remain beneath.
                if bottom_is_king and has_facedown_below:
                    moves.append(Move("tableau", src=src_i, dst=dst_i,
                                      count=count, card=bottom_code))
            elif playable and can_stack_tableau(bottom, top_card):
                moves.append(Move("tableau", src=src_i, dst=dst_i,
                                  count=count, card=bottom_code))

    # 4. Waste top -> tableau
    if w_known:
        w_is_king = w.rank == 13
        w_code = w.code()
        for dst_i in range(7):
            top_card, playable = dst_tops[dst_i]
            if top_card is None:
                if w_is_king:
                    moves.append(Move("waste_to_tableau", dst=dst_i, card=w_code))
            elif playable and can_stack_tableau(w, top_card):
                moves.append(Move("waste_to_tableau", dst=dst_i, card=w_code))

    # 5. Draw (or recycle) from stock
    if state.stock or waste:
        moves.append(Move("draw"))

    return moves
