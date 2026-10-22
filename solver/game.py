"""Klondike game state and move application.

Game variant:
  - 7 tableau columns
  - 4 foundations (build up by suit)
  - stock drawn 3 at a time, recyclable
  - only a King may move onto an empty column
  - tableau builds: alternating color, descending
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .cards import Card, can_stack_foundation, can_stack_tableau


@dataclass(frozen=True)
class Move:
    """A single move.

    kind:
      - "tableau"      : move a run of cards from one column to another
      - "to_foundation": move one card from a column to its foundation
      - "waste_to_tableau"
      - "waste_to_foundation"
      - "foundation_to_tableau"
      - "draw"         : draw from stock (or recycle waste when stock empty)
    """

    kind: str
    src: int | None = None          # source column index (tableau/foundation)
    dst: int | None = None          # destination column index
    count: int = 1                  # number of cards moved (tableau runs)
    card: str | None = None         # code of the primary card (for description)

    def describe(self) -> str:
        if self.kind == "draw":
            return "Draw 3 from the stock"
        if self.kind == "tableau":
            if self.count > 1:
                run = f"the run of {self.count} cards starting at {self.card}"
            else:
                run = self.card
            return f"Move {run} from column {self.src + 1} to column {self.dst + 1}"
        if self.kind == "to_foundation":
            return f"Move {self.card} from column {self.src + 1} to its foundation"
        if self.kind == "waste_to_tableau":
            return f"Move {self.card} from the waste to column {self.dst + 1}"
        if self.kind == "waste_to_foundation":
            return f"Move {self.card} from the waste to its foundation"
        if self.kind == "foundation_to_tableau":
            return f"Move {self.card} from a foundation back to column {self.dst + 1}"
        return self.kind


@dataclass
class GameState:
    # Each tableau column is a list of (Card, face_up: bool)
    tableau: list[list[tuple[Card, bool]]] = field(default_factory=lambda: [[] for _ in range(7)])
    # Foundations keyed by suit -> list of cards (ascending)
    foundations: dict[str, list[Card]] = field(default_factory=lambda: {s: [] for s in ("S", "H", "D", "C")})
    stock: list[Card] = field(default_factory=list)   # face-down, drawn from the end
    waste: list[Card] = field(default_factory=list)    # face-up, top is the end
    draw_count: int = 3
    # Number of cards in the current visible fan (the cards placed by the last
    # draw that haven't been played off yet). 0 before any draw. Up to
    # draw_count; smaller for a ragged tail (1 or 2 leftover cards).
    fan_size: int = 0

    # ---- construction ----

    @classmethod
    def from_layout(
        cls,
        columns: list[list[str]],
        stock: list[str],
        draw_count: int = 3,
        foundations: dict[str, list[str]] | None = None,
        fresh_deal: bool = True,
    ) -> "GameState":
        """Build a fully-known game from card codes.

        columns: 7 lists of card codes, bottom-first.
        stock:   remaining cards as codes, draw order from the front.
        foundations: optional dict suit -> list of card codes already placed.
        fresh_deal: when True (default), the LAST card in each column is
                 face-up and the rest face-down (standard initial deal). When
                 False, ALL cards given are treated as face-up (used for
                 mid-game positions where face-down cards are supplied
                 separately / already resolved).
        """
        if len(columns) != 7:
            raise ValueError("expected 7 tableau columns")
        tableau: list[list[tuple[Card, bool]]] = []
        for col in columns:
            parsed = [Card.parse(c) for c in col]
            if fresh_deal:
                built = [(card, i == len(parsed) - 1) for i, card in enumerate(parsed)]
            else:
                built = [(card, True) for card in parsed]
            tableau.append(built)

        found = cls._build_foundations(foundations)

        gs = cls(
            tableau=tableau,
            foundations=found,
            stock=[Card.parse(c) for c in stock][::-1],  # store reversed so draw pops end
            draw_count=draw_count,
        )
        return gs

    @staticmethod
    def _build_foundations(foundations: dict[str, list[str]] | None) -> dict[str, list[Card]]:
        found: dict[str, list[Card]] = {s: [] for s in ("S", "H", "D", "C")}
        if foundations:
            for suit, codes in foundations.items():
                for code in codes:
                    card = Card.parse(code)
                    if card.suit != suit:
                        raise ValueError(
                            f"{card.code()} does not belong on the {suit} foundation"
                        )
                    found[suit].append(card)
            GameState._validate_foundations(found)
        return found

    @staticmethod
    def _validate_foundations(found: dict[str, list[Card]]) -> None:
        """Each foundation must be a same-suit ascending run starting at the Ace."""
        for suit, pile in found.items():
            for i, card in enumerate(pile):
                if card.rank != i + 1:
                    raise ValueError(
                        f"foundation {suit} must start at Ace and ascend by one; "
                        f"got {[c.code() for c in pile]}"
                    )

    # ---- helpers ----

    def foundation_top(self, suit: str) -> Card | None:
        pile = self.foundations[suit]
        return pile[-1] if pile else None

    def column_face_up_run(self, col: int) -> list[Card]:
        """The movable face-up run of a column, bottom-first.

        A face-up card of unknown identity (a just-flipped '?') is opaque: it
        cannot be moved and it blocks everything below it. So the movable run
        is only the contiguous face-up KNOWN cards above any unknown. If the
        top face-up card is unknown, there is no movable run.
        """
        from .cards import is_unknown

        cells = self.tableau[col]
        run: list[Card] = []
        for card, up in cells:
            if not up:
                run = []  # face-down below; reset (run must be the top face-ups)
                continue
            if is_unknown(card):
                run = []  # unknown blocks; nothing at/below it is movable
                continue
            run.append(card)
        return run

    def column_top_is_unknown(self, col: int) -> bool:
        """True if the column's top card is face-up but of unknown identity."""
        from .cards import is_unknown

        cells = self.tableau[col]
        return bool(cells) and cells[-1][1] and is_unknown(cells[-1][0])

    def clone(self) -> "GameState":
        return GameState(
            tableau=[list(col) for col in self.tableau],
            foundations={s: list(p) for s, p in self.foundations.items()},
            stock=list(self.stock),
            waste=list(self.waste),
            draw_count=self.draw_count,
            fan_size=self.fan_size,
        )

    def is_won(self) -> bool:
        return all(len(p) == 13 for p in self.foundations.values())

    # ---- canonical key for search dedupe ----

    def key(self) -> tuple:
        tab = tuple(
            tuple((c.rank, c.suit, up) for c, up in col)
            for col in self.tableau
        )
        found = tuple((s, len(self.foundations[s])) for s in ("S", "H", "D", "C"))
        stock = tuple((c.rank, c.suit) for c in self.stock)
        waste = tuple((c.rank, c.suit) for c in self.waste)
        return (tab, found, stock, waste)
