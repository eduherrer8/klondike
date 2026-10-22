"""Card model for Klondike solitaire.

Cards are represented as compact strings like "AS" (Ace of Spades),
"10H" / "TH" (Ten of Hearts), "KC" (King of Clubs).

Internally a card is a (rank, suit) tuple:
  rank: 1..13  (A=1, J=11, Q=12, K=13)
  suit: one of "S", "H", "D", "C"
"""

from __future__ import annotations

SUITS = ("S", "H", "D", "C")
RED_SUITS = frozenset({"H", "D"})
BLACK_SUITS = frozenset({"S", "C"})

RANK_TO_STR = {
    1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
    8: "8", 9: "9", 10: "10", 11: "J", 12: "Q", 13: "K",
}

STR_TO_RANK = {
    "A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "10": 10, "T": 10, "J": 11, "Q": 12, "K": 13,
}

SUIT_SYMBOLS = {"S": "\u2660", "H": "\u2665", "D": "\u2666", "C": "\u2663"}


class Card:
    __slots__ = ("rank", "suit")

    def __init__(self, rank: int, suit: str):
        if not 1 <= rank <= 13:
            raise ValueError(f"invalid rank: {rank}")
        if suit not in SUITS:
            raise ValueError(f"invalid suit: {suit}")
        self.rank = rank
        self.suit = suit

    @property
    def is_red(self) -> bool:
        return self.suit in RED_SUITS

    @property
    def is_black(self) -> bool:
        return self.suit in BLACK_SUITS

    @classmethod
    def parse(cls, text: str) -> "Card":
        text = text.strip().upper()
        if len(text) < 2:
            raise ValueError(f"cannot parse card: {text!r}")
        suit = text[-1]
        rank_str = text[:-1]
        if rank_str not in STR_TO_RANK:
            raise ValueError(f"unknown rank in {text!r}")
        if suit not in SUITS:
            raise ValueError(f"unknown suit in {text!r}")
        return cls(STR_TO_RANK[rank_str], suit)

    def code(self) -> str:
        """Canonical code, e.g. 'AS', '10H', 'KC'."""
        return f"{RANK_TO_STR[self.rank]}{self.suit}"

    def pretty(self) -> str:
        return f"{RANK_TO_STR[self.rank]}{SUIT_SYMBOLS[self.suit]}"

    def __eq__(self, other) -> bool:
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))

    def __repr__(self) -> str:
        return f"Card({self.code()})"


def full_deck() -> list[Card]:
    return [Card(rank, suit) for suit in SUITS for rank in range(1, 14)]


# Sentinel used in raw layouts to mark a face-down card of unknown identity.
UNKNOWN_CODE = "?"


class _Unknown:
    """Sentinel for a card whose identity is not known.

    Used in the strategy planner's model where face-down cards are opaque:
    we know a card occupies a slot but not which card. An Unknown is never
    equal to a real Card and cannot be stacked or sent to a foundation.
    """

    __slots__ = ()
    rank = None
    suit = None

    @property
    def is_red(self) -> bool:  # pragma: no cover - should never be queried
        return False

    @property
    def is_black(self) -> bool:  # pragma: no cover
        return False

    def code(self) -> str:
        return UNKNOWN_CODE

    def pretty(self) -> str:
        return UNKNOWN_CODE

    def __eq__(self, other) -> bool:
        return other is self or isinstance(other, _Unknown)

    def __hash__(self) -> int:
        return hash("__unknown_card__")

    def __repr__(self) -> str:
        return "Unknown()"


UNKNOWN = _Unknown()


def is_unknown(card) -> bool:
    return isinstance(card, _Unknown)


def can_stack_tableau(moving: Card, onto: Card) -> bool:
    """Alternating color, descending by one. e.g. red 6 onto black 7."""
    return moving.rank + 1 == onto.rank and moving.is_red != onto.is_red


def can_stack_foundation(moving: Card, top: Card | None, foundation_suit: str | None) -> bool:
    """Same suit, ascending by one. Ace onto empty foundation."""
    if top is None:
        return moving.rank == 1
    return moving.suit == top.suit and moving.rank == top.rank + 1
