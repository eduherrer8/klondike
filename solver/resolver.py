"""Resolve a mid-game layout with hidden (face-down) cards.

A mid-game position is described with:
  - columns: 7 lists of tokens, bottom-first. A token is either a card code
    (face-up) or "?" (a face-down card of unknown identity). Face-down cards
    must come BEFORE the face-up ones in a column (they are physically below).
  - foundations: current pile per suit (ascending from the Ace).
  - stock: the fixed cycling stock sequence, draw order front-first.
  - waste: cards currently fanned out of the stock (optional), bottom-first,
    top card last.

Because the game uses a full 52-card deck, the identities of the "?" cards are
determined by elimination: they are exactly the cards not visible anywhere
else. We don't know their ORDER within the hidden stacks, so the solver
enumerates arrangements (see solver.solve_midgame).
"""

from __future__ import annotations

from dataclasses import dataclass

from .cards import UNKNOWN_CODE, Card, full_deck
from .game import GameState


@dataclass
class MidGameLayout:
    # Per column: list of (code_or_None, face_up). None means a face-down
    # unknown slot to be filled from `unknown_pool`.
    columns: list[list[tuple[str | None, bool]]]
    foundations: dict[str, list[str]]
    stock: list[str]
    waste: list[str]
    draw_count: int
    unknown_pool: list[Card]          # the cards that must fill the "?" slots
    unknown_slots: list[tuple[int, int]]  # (col_index, position_in_column) per "?"


def parse_midgame(
    columns: list[list[str]],
    stock: list[str],
    foundations: dict[str, list[str]] | None = None,
    waste: list[str] | None = None,
    draw_count: int = 3,
) -> MidGameLayout:
    foundations = foundations or {}
    waste = waste or []

    parsed_columns: list[list[tuple[str | None, bool]]] = []
    unknown_slots: list[tuple[int, int]] = []
    known_codes: set[str] = set()

    def register(code: str) -> None:
        card = Card.parse(code)
        norm = card.code()
        if norm in known_codes:
            raise ValueError(f"duplicate card: {norm}")
        known_codes.add(norm)

    for col_i, col in enumerate(columns):
        built: list[tuple[str | None, bool]] = []
        seen_face_up = False
        for pos, tok in enumerate(col):
            tok = tok.strip()
            if tok == UNKNOWN_CODE:
                if seen_face_up:
                    raise ValueError(
                        f"column {col_i + 1}: face-down '?' cannot appear above a face-up card"
                    )
                built.append((None, False))
                unknown_slots.append((col_i, pos))
            else:
                register(tok)
                built.append((Card.parse(tok).code(), True))
                seen_face_up = True
        parsed_columns.append(built)

    for suit, codes in foundations.items():
        for code in codes:
            register(code)
    # Validate foundations here (suit correctness + Ace-up ascending run) so
    # bad foundations surface as a clean parse error, not a later crash.
    GameState._build_foundations(foundations)
    for code in stock:
        register(code)
    for code in waste:
        register(code)

    # The unknown pool = full deck minus everything known.
    deck = full_deck()
    pool = [c for c in deck if c.code() not in known_codes]

    if len(pool) != len(unknown_slots):
        raise ValueError(
            f"found {len(unknown_slots)} face-down slot(s) but {len(pool)} "
            f"card(s) are unaccounted for; they must match. Check your inputs "
            f"total 52 unique cards."
        )

    return MidGameLayout(
        columns=parsed_columns,
        foundations=foundations,
        stock=stock,
        waste=waste,
        draw_count=draw_count,
        unknown_pool=pool,
        unknown_slots=unknown_slots,
    )


def build_opaque_state(layout: MidGameLayout) -> GameState:
    """Build a GameState where face-down cards are OPAQUE (UNKNOWN sentinel).

    This is the state the strategy planner reasons about: known cards are
    concrete, hidden cards are unknown placeholders that cannot be moved and
    block whatever is beneath them. The top card of a column, if it was the
    only (face-down) card, is treated as exposed/unknown (face-up unknown).
    """
    from .cards import UNKNOWN

    tableau: list[list[tuple[Card, bool]]] = []
    for col_i, col in enumerate(layout.columns):
        built: list[tuple[Card, bool]] = []
        last = len(col) - 1
        for pos, (code, face_up) in enumerate(col):
            if code is None:
                # Unknown. If it's the exposed top card, it's a face-up unknown
                # (a card you'd see is a card you'd have entered, but a lone
                # hidden top is exposed in reality) -> mark face-up unknown.
                built.append((UNKNOWN, pos == last))
            else:
                built.append((Card.parse(code), True))
        tableau.append(built)

    found = GameState._build_foundations(layout.foundations)
    return GameState(
        tableau=tableau,
        foundations=found,
        stock=[Card.parse(c) for c in layout.stock][::-1],
        waste=[Card.parse(c) for c in layout.waste],
        draw_count=layout.draw_count,
    )


def build_state(layout: MidGameLayout, assignment: dict[tuple[int, int], Card]) -> GameState:
    """Build a concrete GameState from the layout and an assignment of the
    unknown slots to specific cards. Face-down slots become face-down cards.

    Normalization: the top card of a column is always face-up in real
    solitaire, so if a column's last card was marked face-down it is flipped
    up here.
    """
    tableau: list[list[tuple[Card, bool]]] = []
    for col_i, col in enumerate(layout.columns):
        built: list[tuple[Card, bool]] = []
        last = len(col) - 1
        for pos, (code, face_up) in enumerate(col):
            if code is None:
                card = assignment[(col_i, pos)]
                built.append((card, pos == last))  # exposed bottom card is face-up
            else:
                built.append((Card.parse(code), True))
        tableau.append(built)

    found = GameState._build_foundations(layout.foundations)

    gs = GameState(
        tableau=tableau,
        foundations=found,
        stock=[Card.parse(c) for c in layout.stock][::-1],
        waste=[Card.parse(c) for c in layout.waste],
        draw_count=layout.draw_count,
    )
    return gs
