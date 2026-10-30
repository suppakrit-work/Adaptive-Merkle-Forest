"""Distributed Index -- Section III-A 4), Table I, Definition 2.

One table with one row per tree, live or retired:

    own | tid | [lo, hi) | prov | r | n | v | sigma

keyed by (own, tid).  An owner is the only writer of the rows of its own
trees and sends every new row to every provider.  Each provider's DHT node
keeps an identical copy and accepts a row only if the signature verifies under
the public key of the owner the row names, the range lies in that owner's
identifier space, and the version is higher than the one it holds for that
owner and tree identifier.  A split or a merge retires the rows it replaces,
each written once more with a higher version and an empty range.  The rows of
one operation travel as one batch, which a node applies whole or not at all.

Forest digest D: the root of a tree of the canonical shape over the owner's T
live rows in tree-identifier order, signed with a version of its own and sent
with the rows.  A reader checks its row against D with a path of
ceil(log2 T) hashes; an auditor that takes all the rows recomputes D.

Prototype decisions: identifiers are compared as 65-byte big-endian integers;
a range is half-open [lo, hi); the tree opened for a type tag covers every
identifier that starts with own || type tag.  Signatures: Ed25519 over a
fixed-width serialization with a domain prefix.  A row's leaf in D's tree is
SHA-256("amf/row-leaf" || payload).
"""
import hashlib
import os
from bisect import bisect_left, bisect_right, insort
from dataclasses import dataclass, replace
from operator import itemgetter
from types import MappingProxyType

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from amf.common.crypto import HMAC_LEN, ID_LEN, OWN_LEN, owner_tag
from amf.common.merkle import MerkleTree, verify_path

TREE_ID_LEN = 16
BOUND_LEN = ID_LEN + 1              # an upper bound can be 2^(8*65)
SIG_LEN = 64
ROW_DOMAIN = b"amf/index-row/v2"
DIGEST_DOMAIN = b"amf/forest-digest/v1"
ROW_LEAF_DOMAIN = b"amf/row-leaf"


def id_to_int(obj_id: bytes) -> int:
    if len(obj_id) != ID_LEN:
        raise ValueError("not a 65-byte identifier")
    return int.from_bytes(obj_id, "big")


def owner_space(own: bytes) -> tuple[int, int]:
    """Every identifier that starts with this owner's tag."""
    if len(own) != OWN_LEN:
        raise ValueError("the owner's tag is 32 bytes")
    shift = 8 * (ID_LEN - OWN_LEN)
    v = int.from_bytes(own, "big")
    return v << shift, (v + 1) << shift


def tag_range(own: bytes, type_tag: int) -> tuple[int, int]:
    """The identifier range of one type of one owner: every id starting with own || type tag."""
    if not 0 <= type_tag <= 255:
        raise ValueError("a type tag must fit in one byte")
    prefix = int.from_bytes(own + bytes([type_tag]), "big")
    shift = 8 * HMAC_LEN
    return prefix << shift, (prefix + 1) << shift


def new_tree_id() -> bytes:
    """An identifier never used before (128 random bits)."""
    return os.urandom(TREE_ID_LEN)


@dataclass(frozen=True)
class IndexRow:
    own: bytes               # owner's tag, the hash of its public key
    tree_id: bytes
    lo: int                  # inclusive
    hi: int                  # exclusive; lo == hi means the row is retired
    provider: str
    root: bytes
    count: int
    version: int
    signature: bytes = b""

    @property
    def key(self) -> tuple[bytes, bytes]:
        return self.own, self.tree_id

    @property
    def retired(self) -> bool:
        return self.lo == self.hi

    def covers(self, id_int: int) -> bool:
        return self.lo <= id_int < self.hi

    def payload(self) -> bytes:
        """Fixed-width serialization of the other seven fields, what the owner signs."""
        prov = self.provider.encode("utf-8")
        if len(prov) > 255:
            raise ValueError("provider name too long")
        return (ROW_DOMAIN + self.own + self.tree_id
                + self.lo.to_bytes(BOUND_LEN, "big") + self.hi.to_bytes(BOUND_LEN, "big")
                + bytes([len(prov)]) + prov + self.root
                + self.count.to_bytes(8, "big") + self.version.to_bytes(8, "big"))


def row_leaf(row: IndexRow) -> bytes:
    """The row's leaf in the tree whose root is D."""
    return hashlib.sha256(ROW_LEAF_DOMAIN + row.payload()).digest()


# -- the owner's signing side (Key/Signing Manager) ---------------------------

def generate_signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def sign_row(sk: Ed25519PrivateKey, row: IndexRow) -> IndexRow:
    return replace(row, signature=sk.sign(row.payload()))


def verify_row(pk, row: IndexRow) -> bool:
    if len(row.signature) != SIG_LEN or row.own != owner_tag(pk):
        return False
    try:
        pk.verify(row.signature, row.payload())
        return True
    except InvalidSignature:
        return False


def retire(row: IndexRow) -> IndexRow:
    """The row written once more with a higher version and an empty range (unsigned yet)."""
    return replace(row, hi=row.lo, version=row.version + 1, signature=b"")


# -- the forest digest D ---------------------------------------------------------

@dataclass(frozen=True)
class ForestDigest:
    own: bytes
    root: bytes              # canonical-shape Merkle root over the live rows in tree-identifier order
    count: int               # T, the number of live rows
    version: int
    signature: bytes = b""

    def payload(self) -> bytes:
        return (DIGEST_DOMAIN + self.own + self.root
                + self.count.to_bytes(8, "big") + self.version.to_bytes(8, "big"))


def live_in_order(rows) -> list[IndexRow]:
    """The live rows in tree-identifier order: the leaves of D's tree."""
    return sorted((r for r in rows if not r.retired), key=lambda r: r.tree_id)


def digest_tree(rows) -> tuple[MerkleTree, list[IndexRow]]:
    ordered = live_in_order(rows)
    return MerkleTree([row_leaf(r) for r in ordered]), ordered


def sign_digest(sk: Ed25519PrivateKey, d: ForestDigest) -> ForestDigest:
    return replace(d, signature=sk.sign(d.payload()))


def verify_digest(pk, d: ForestDigest) -> bool:
    if d is None or len(d.signature) != SIG_LEN or d.own != owner_tag(pk):
        return False
    try:
        pk.verify(d.signature, d.payload())
        return True
    except InvalidSignature:
        return False


def verify_row_in_digest(d: ForestDigest, row: IndexRow, i: int, path: list) -> bool:
    """Does the row sit at position i under D?  ceil(log2 T) hashes."""
    return not row.retired and verify_path(d.root, d.count, i, row_leaf(row), path)


# -- a DHT node: one provider's copy of the table -----------------------------

class DHTNode:
    """One provider's copy of the table, with the rows of each owner kept apart.

    Each owner's live ranges are kept sorted, so a lookup is a binary search
    over that owner's rows and a batch touches only the rows it carries; the
    work does not grow with the number of other owners.
    """

    def __init__(self, keys=None):
        self.keys: dict[bytes, object] = {}          # own -> public key (registered at Phase 1)
        self.digests: dict[bytes, ForestDigest] = {}
        self.rows = {}                               # an empty table and its index
        for pk in (keys or []):
            self.register(pk)

    @property
    def rows(self):
        """(own, tid) -> highest version seen, read-only; assigning a whole table rebuilds the index."""
        return MappingProxyType(self._rows)

    @rows.setter
    def rows(self, rows: dict) -> None:
        self._rows: dict[tuple, IndexRow] = {}
        self._by_own: dict[bytes, dict[bytes, IndexRow]] = {}   # own -> tid -> row
        self._spans: dict[bytes, list[tuple[int, bytes]]] = {}  # own -> (lo, tid) of live rows, sorted
        for row in rows.values():
            self._put(row)

    def register(self, pk) -> bytes:
        own = owner_tag(pk)
        self.keys[own] = pk
        return own

    def _acceptable(self, row: IndexRow, held: IndexRow | None) -> bool:
        pk = self.keys.get(row.own)
        if pk is None or not verify_row(pk, row):
            return False
        lo, hi = owner_space(row.own)
        if not (lo <= row.lo <= row.hi <= hi):
            return False
        return held is None or row.version > held.version

    def _put(self, row: IndexRow) -> None:
        prev = self._rows.get(row.key)
        spans = self._spans.setdefault(row.own, [])
        if prev is not None and not prev.retired:
            del spans[bisect_left(spans, (prev.lo, prev.tree_id))]
        if not row.retired:
            insort(spans, (row.lo, row.tree_id))
        self._rows[row.key] = row
        self._by_own.setdefault(row.own, {})[row.tree_id] = row

    def accept(self, row: IndexRow) -> bool:
        """Store the row only if the signature verifies, the range is the owner's and the version is higher."""
        return self.accept_batch([row])

    def accept_batch(self, rows, digest: ForestDigest | None = None) -> bool:
        """The rows of one operation, applied whole or not at all, with the owner's new digest."""
        staged = {}
        for row in rows:
            held = staged[row.key] if row.key in staged else self._rows.get(row.key)
            if not self._acceptable(row, held):
                return False
            staged[row.key] = row
        if digest is not None:
            pk, prev = self.keys.get(digest.own), self.digests.get(digest.own)
            if pk is None or not verify_digest(pk, digest) or (prev and digest.version <= prev.version):
                return False
            self.digests[digest.own] = digest
        for row in staged.values():
            self._put(row)
        return True

    def get(self, own: bytes, tree_id: bytes) -> IndexRow | None:
        return self._rows.get((own, tree_id))

    def digest(self, own: bytes) -> ForestDigest | None:
        return self.digests.get(own)

    def tree_ids(self, own: bytes) -> list[bytes]:
        """Every tree identifier the node holds for this owner, live or retired."""
        return list(self._by_own.get(own, {}))

    def live_rows(self, own: bytes | None = None) -> list[IndexRow]:
        rows = self._rows.values() if own is None else self._by_own.get(own, {}).values()
        return [r for r in rows if not r.retired]

    def lookup(self, obj_id: bytes) -> IndexRow | None:
        """Routing: which live tree holds this identifier?  A binary search over the owner's live ranges."""
        v = id_to_int(obj_id)
        own = obj_id[:OWN_LEN]
        spans = self._spans.get(own, [])
        i = bisect_right(spans, v, key=itemgetter(0)) - 1
        if i < 0:
            return None
        row = self._by_own[own][spans[i][1]]
        return row if row.covers(v) else None

    def row_with_path(self, own: bytes, tree_id: bytes):
        """The row and its path to D: (row, position among the live rows, T, sibling hashes)."""
        row = self.get(own, tree_id)
        if row is None or row.retired:
            return row, None, 0, []
        tree, ordered = digest_tree(self.live_rows(own))
        i = [r.key for r in ordered].index(row.key)
        return row, i, len(ordered), tree.path(i)

    def route(self, obj_id: bytes):
        """What a reader asks a node, in one answer from one view of the table: the live row that covers
        the identifier, its position, T and path to D, and the owner's D as this node holds it."""
        own = obj_id[:OWN_LEN]
        row = self.lookup(obj_id)
        if row is None:
            return None, None, 0, [], self.digest(own)
        _row, i, t, path = self.row_with_path(own, row.tree_id)
        return row, i, t, path, self.digest(own)


def freshest(rows) -> IndexRow | None:
    """What a reader other than the owner does: ask several nodes, keep the highest version."""
    rows = [r for r in rows if r is not None]
    return max(rows, key=lambda r: r.version) if rows else None


def freshest_signed(pk, rows, key) -> IndexRow | None:
    """The highest version among the rows for key (own, tid) whose signature verifies under pk, so a node
    that hands out a forged row with a high version only costs one signature check."""
    for r in sorted((r for r in rows if r is not None and r.key == key), key=lambda r: r.version, reverse=True):
        if verify_row(pk, r):
            return r
    return None
