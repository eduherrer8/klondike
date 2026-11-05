"""Move generation and application for Klondike."""

from __future__ import annotations

from .cards import UNKNOWN, Card, can_stack_foundation, can_stack_tableau, is_unknown
from .game import (
    GameState,
    Move,
    _FOUND_SUIT_IDX,
    _Z_FOUND,
    _Z_STOCK,
    _Z_TAB,
    _Z_WASTE,
    _cell_slot,
)


def _tab_cell_hash(col_i: int, pos: int, card: "Card", up: bool) -> int:
    """Zobrist contribution of a single tableau cell at (col_i, pos)."""
    return _Z_TAB[col_i][pos][_cell_slot(card.kid, up)]


def _flip_top_delta(col_i: int, col: list[tuple[Card, bool]]) -> int:
    """Flip the new top card of `col` face-up in place and return the Zobrist
    delta for the running hash. `col` is a fresh, owned (shortened) list; the
    flipped position is its last index. Returns 0 when there is nothing to flip.
    """
    if col and not col[-1][1]:
        card, _ = col[-1]
        pos = len(col) - 1
        before = _tab_cell_hash(col_i, pos, card, False)
        col[-1] = (card, True)
        after = _tab_cell_hash(col_i, pos, card, True)
        return before ^ after
    return 0


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
    h = state.zobrist()  # parent hash; each branch XORs only what it touches

    if kind == "draw":
        if not state.stock:
            # recycle waste back into stock (preserving order). stock was empty
            # (no contribution); waste cells leave, new stock cells arrive.
            for pos, c in enumerate(state.waste):
                h ^= _Z_WASTE[pos][c.kid]
            new_stock = state.waste[::-1]
            for pos, c in enumerate(new_stock):
                h ^= _Z_STOCK[pos][c.kid]
            return GameState(
                tableau=tab, foundations=state.foundations,
                stock=new_stock, waste=[], draw_count=state.draw_count,
                fan_size=0, zkey=h,
            )
        n = min(state.draw_count, len(state.stock))
        new_stock = state.stock[:-n]
        drawn = state.stock[-1:-n - 1:-1]  # last n, in pop order (end first)
        # Stock loses its last n positions; waste gains n at the tail.
        base = len(state.stock) - n
        for i in range(n):
            c = state.stock[base + i]
            h ^= _Z_STOCK[base + i][c.kid]
        wbase = len(state.waste)
        for i, c in enumerate(drawn):
            h ^= _Z_WASTE[wbase + i][c.kid]
        return GameState(
            tableau=tab, foundations=state.foundations,
            stock=new_stock, waste=state.waste + drawn,
            draw_count=state.draw_count, fan_size=n, zkey=h,
        )

    if kind == "tableau":
        new_tab = list(tab)
        src_col = tab[move.src]
        count = move.count
        src_len = len(src_col)
        moving = src_col[-count:]
        shortened = src_col[:-count]
        # XOR out the moved cells from the source column (at their old
        # positions), then XOR them back in on the destination column (at their
        # new positions). Orientation is unchanged (a movable run is face-up).
        src_start = src_len - count
        for i, (c, up) in enumerate(moving):
            h ^= _tab_cell_hash(move.src, src_start + i, c, up)
        h ^= _flip_top_delta(move.src, shortened)
        new_tab[move.src] = shortened
        dst_col = tab[move.dst]
        dst_base = len(dst_col)
        for i, (c, up) in enumerate(moving):
            h ^= _tab_cell_hash(move.dst, dst_base + i, c, up)
        new_tab[move.dst] = dst_col + moving
        return GameState(
            tableau=new_tab, foundations=state.foundations,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size, zkey=h,
        )

    if kind == "to_foundation":
        new_tab = list(tab)
        src_col = list(tab[move.src])
        top_pos = len(src_col) - 1
        card, up = src_col[-1]
        # Remove the top card cell from the source column.
        h ^= _tab_cell_hash(move.src, top_pos, card, up)
        src_col.pop()
        h ^= _flip_top_delta(move.src, src_col)
        new_tab[move.src] = src_col
        # Foundation length for this suit grows by one.
        fi = _FOUND_SUIT_IDX[card.suit]
        old_len = len(state.foundations[card.suit])
        h ^= _Z_FOUND[fi][old_len]
        h ^= _Z_FOUND[fi][old_len + 1]
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit] + [card]
        return GameState(
            tableau=new_tab, foundations=new_found,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size, zkey=h,
        )

    if kind == "waste_to_tableau":
        card = state.waste[-1]
        wpos = len(state.waste) - 1
        # Waste top (face-up known card) leaves its position.
        h ^= _Z_WASTE[wpos][card.kid]
        dst_col = tab[move.dst]
        dst_pos = len(dst_col)
        h ^= _tab_cell_hash(move.dst, dst_pos, card, True)
        new_tab = list(tab)
        new_tab[move.dst] = dst_col + [(card, True)]
        return GameState(
            tableau=new_tab, foundations=state.foundations,
            stock=state.stock, waste=state.waste[:-1],
            draw_count=state.draw_count,
            fan_size=max(0, state.fan_size - 1), zkey=h,
        )

    if kind == "waste_to_foundation":
        card = state.waste[-1]
        wpos = len(state.waste) - 1
        h ^= _Z_WASTE[wpos][card.kid]
        fi = _FOUND_SUIT_IDX[card.suit]
        old_len = len(state.foundations[card.suit])
        h ^= _Z_FOUND[fi][old_len]
        h ^= _Z_FOUND[fi][old_len + 1]
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit] + [card]
        return GameState(
            tableau=tab, foundations=new_found,
            stock=state.stock, waste=state.waste[:-1],
            draw_count=state.draw_count,
            fan_size=max(0, state.fan_size - 1), zkey=h,
        )

    if kind == "foundation_to_tableau":
        card = Card.parse(move.card)
        fi = _FOUND_SUIT_IDX[card.suit]
        old_len = len(state.foundations[card.suit])
        h ^= _Z_FOUND[fi][old_len]
        h ^= _Z_FOUND[fi][old_len - 1]
        new_found = dict(state.foundations)
        new_found[card.suit] = state.foundations[card.suit][:-1]
        dst_col = tab[move.dst]
        dst_pos = len(dst_col)
        h ^= _tab_cell_hash(move.dst, dst_pos, card, True)
        new_tab = list(tab)
        new_tab[move.dst] = dst_col + [(card, True)]
        return GameState(
            tableau=new_tab, foundations=new_found,
            stock=state.stock, waste=state.waste,
            draw_count=state.draw_count, fan_size=state.fan_size, zkey=h,
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
