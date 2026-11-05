"""Klondike game state and move application.

Game variant:
  - 7 tableau columns
  - 4 foundations (build up by suit)
  - stock drawn 3 at a time, recyclable
  - only a King may move onto an empty column
  - tableau builds: alternating color, descending
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field, replace

from .cards import Card, can_stack_foundation, can_stack_tableau

# Reserved key-id sentinels for a face-up / face-down UNKNOWN card. Chosen
# outside the real kid range (4..55) and distinct from each other so an unknown
# never collides with a real card and the two orientations stay distinguishable
# (matching the old (None, None, up) tuple encoding).
_UNKNOWN_UP_KID = 0
_UNKNOWN_DOWN_KID = 100


# ---------------------------------------------------------------------------
# Incremental Zobrist hashing (Task 2 of docs/plan-solver-speed-2.md)
#
# GameState.key() rebuilds a full nested tuple of ints for EVERY generated
# state (~25% of solve time). The solver only needs a hashable identity for
# dedupe, so instead of rebuilding the whole key we maintain a single 64-bit
# integer ("zkey") that is a pure function of position and is updated by
# XOR-ing in/out only the cells a move touches (see engine.apply_move).
#
# The hash captures EXACTLY what key() captures, so equal positions hash equal
# and different positions almost surely differ:
#   * per-column card identity + orientation, at its position in the column
#   * foundation pile LENGTHS (per suit)
#   * full stock order
#   * full waste order
#
# The constant table is a MODULE-LEVEL CONSTANT seeded with a FIXED seed, so it
# is identical in every process (including spawned pool workers that unpickle
# GameStates) and across runs. This is the single most important determinism
# property: a per-process / unseeded table would silently break dedupe
# agreement between workers. Do NOT regenerate this per state or per process.
# ---------------------------------------------------------------------------

_ZOBRIST_SEED = 0x6B6C6F6E64696B65  # "klondike" in ascii, fixed forever
_MASK64 = (1 << 64) - 1

# Generous static bounds (a real Klondike position never exceeds these):
#   tableau: 7 columns, a column holds at most 13 face-up (K..A) + up to 6
#     face-down beneath = 19; 24 is a safe ceiling.
#   stock/waste: 24 cards total in the stock+waste cycle; 24 each is safe.
#   cell "slot": encodes (card, orientation) into a small index, see
#     _cell_slot below; max index is 2 + 55*2 + 1 = 113, so 128 is safe.
#   kid: 0..55 (real cards are 4..55); 56 covers it.
#   foundation length: 0..13 -> 14 entries.
_TAB_COLS = 7
_MAX_COL_DEPTH = 24
_MAX_PILE = 24
_N_SLOTS = 128
_N_KID = 56
_N_FOUND_LEN = 14


def _cell_slot(kid: int | None, up: bool) -> int:
    """Map a tableau cell (card identity, orientation) to a small slot index.

    Mirrors key()'s per-cell encoding: a face-up UNKNOWN, a face-down UNKNOWN,
    and each real card in each orientation get a distinct index. This is the
    index into the per-(column, position) Zobrist row.
    """
    if kid is None:
        return 0 if up else 1
    return 2 + kid * 2 + (0 if up else 1)


def _build_zobrist_tables():
    rng = _random.Random(_ZOBRIST_SEED)

    def rand():
        return rng.getrandbits(64)

    # Z_TAB[col][pos][slot]: contribution of a tableau cell by position.
    z_tab = [[[rand() for _ in range(_N_SLOTS)]
              for _ in range(_MAX_COL_DEPTH)]
             for _ in range(_TAB_COLS)]
    # Z_FOUND[suit_idx][length]: contribution of a foundation pile's length.
    z_found = [[rand() for _ in range(_N_FOUND_LEN)] for _ in range(4)]
    # Z_STOCK[pos][kid] / Z_WASTE[pos][kid]: order-sensitive pile contributions.
    z_stock = [[rand() for _ in range(_N_KID)] for _ in range(_MAX_PILE)]
    z_waste = [[rand() for _ in range(_N_KID)] for _ in range(_MAX_PILE)]
    return z_tab, z_found, z_stock, z_waste


_Z_TAB, _Z_FOUND, _Z_STOCK, _Z_WASTE = _build_zobrist_tables()

# Suit -> foundation index, matching the (S, H, D, C) order used by key().
_FOUND_SUIT_IDX = {"S": 0, "H": 1, "D": 2, "C": 3}


def compute_zkey(state: "GameState") -> int:
    """Compute the full Zobrist hash of a position from scratch.

    Used once for a freshly built / determinized / cloned state; thereafter
    engine.apply_move maintains it incrementally. This is a PURE function of
    position capturing exactly what key() captures, so it must agree with an
    incrementally-maintained value for the same position.
    """
    h = 0
    for col_i, col in enumerate(state.tableau):
        ztab_col = _Z_TAB[col_i]
        for pos, (card, up) in enumerate(col):
            kid = card.kid
            h ^= ztab_col[pos][_cell_slot(kid, up)]
    f = state.foundations
    h ^= _Z_FOUND[0][len(f["S"])]
    h ^= _Z_FOUND[1][len(f["H"])]
    h ^= _Z_FOUND[2][len(f["D"])]
    h ^= _Z_FOUND[3][len(f["C"])]
    for pos, c in enumerate(state.stock):
        h ^= _Z_STOCK[pos][c.kid]
    for pos, c in enumerate(state.waste):
        h ^= _Z_WASTE[pos][c.kid]
    return h


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
    # Incrementally-maintained 64-bit Zobrist hash of this position (Task 2).
    # A plain int (picklable, carried through clone). engine.apply_move supplies
    # it for derived states; for a freshly-built or directly-mutated state it is
    # None until zobrist() computes it from scratch on first use. Never rely on
    # a cached value after mutating tableau/stock/waste/foundations in place —
    # zobrist() is only computed for states the solver actually searches (the
    # solver recomputes the start state's hash from scratch at the top of the
    # search), and apply_move keeps it exact for every child.
    zkey: int | None = None

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

    def reveal_top(self, col: int, card: Card) -> None:
        """Pin the real identity of a just-flipped face-up unknown top card.

        This is the ONE sanctioned way to mutate a (possibly long-lived) state
        in place: it replaces the face-up UNKNOWN top of `col` with the real
        `card` AND invalidates the cached Zobrist hash in a single step, so the
        two can never drift apart. A state derived via engine.apply_move carries
        an incrementally-maintained `zkey`; editing a cell without clearing it
        leaves the hash stale, which corrupts the solver's dedupe and can make
        winnable positions look like dead ends. Routing reveals through here
        means callers can't forget that invalidation.

        Raises ValueError if the column top is not a face-up unknown awaiting
        reveal (the only legitimate target).
        """
        from .cards import is_unknown

        cells = self.tableau[col]
        if not cells or not cells[-1][1] or not is_unknown(cells[-1][0]):
            raise ValueError(
                f"column {col} top is not a face-up unknown card to reveal"
            )
        cells[-1] = (card, True)
        self.zkey = None  # in-place mutation: force zobrist() to recompute

    def clone(self) -> "GameState":
        return GameState(
            tableau=[list(col) for col in self.tableau],
            foundations={s: list(p) for s, p in self.foundations.items()},
            stock=list(self.stock),
            waste=list(self.waste),
            draw_count=self.draw_count,
            fan_size=self.fan_size,
            zkey=self.zkey,
        )

    def is_won(self) -> bool:
        return all(len(p) == 13 for p in self.foundations.values())

    # ---- canonical key for search dedupe ----

    def zobrist(self) -> int:
        """64-bit Zobrist hash of this position (the solver's dedupe key).

        Returns the incrementally-maintained value if present (set by
        engine.apply_move for derived states, or carried through clone), else
        computes it from scratch and caches it. Computing from scratch is only
        needed for a freshly-built or directly-mutated state; the solver does it
        once for the start state, after which every child carries an exact
        incremental value.

        This is a PURE function of position identical to what key() encodes, so
        equal positions share a hash and the dedupe/determinism contracts hold.
        """
        if self.zkey is None:
            self.zkey = compute_zkey(self)
        return self.zkey

    def key(self) -> tuple:
        """Canonical, hashable identity of this state for search dedupe.

        Each card is encoded by its precomputed compact id (`Card.kid`, a small
        int unique per card), with 0 as the sentinel for a face-up UNKNOWN. A
        face-up card is tagged by making its entry positive; a face-down card
        negates it, so (identity, face_up) collapses into a single int and we
        avoid building per-card (rank, suit, up) tuples in the hot path.

        Determinism: the encoding is a pure function of card identity and
        orientation, independent of object identity or dict ordering, so equal
        positions always produce equal keys and different positions differ.
        """
        # Encode each column inline (no per-column helper call) as a tuple of
        # small ints: +kid face-up, -kid face-down, reserved sentinels for an
        # UNKNOWN. This runs in the interpreter's fast comprehension path.
        tab = tuple([
            tuple([
                (c.kid if up else -c.kid) if c.kid is not None
                else (_UNKNOWN_UP_KID if up else _UNKNOWN_DOWN_KID)
                for c, up in col
            ])
            for col in self.tableau
        ])
        f = self.foundations
        found = (len(f["S"]), len(f["H"]), len(f["D"]), len(f["C"]))
        stock = tuple([c.kid for c in self.stock])
        waste = tuple([c.kid for c in self.waste])
        return (tab, found, stock, waste)
