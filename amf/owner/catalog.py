"""The owner's catalog -- the state of the Verification Manager (III-A 1 b, Table I).

The catalog maps file names to identifiers and leaf hashes and keeps the
identifiers in identifier order, so the owner can, for any range [lo, hi):

* count the identifiers in it                      (n of every announced tree)
* list their leaf values in identifier order       (to compute the root itself,
                                                    n - 1 hash evaluations)
* turn a position inside a tree into an identifier, and the identifier into a
  file name                                        (naming the object, Phase 5)

The structure is a treap keyed by identifier whose nodes carry the size and
the key bounds of their subtree: insert, delete, rank, select and range count
cost O(log n) expected.

The encrypted copy at every provider is kept entry by entry: each change sends
only the new entry, encrypted on its own under a key derived from MK and
tagged with the version of D; on recovery the highest version of each entry
wins.  The type table travels the same way.
"""
import json
import os
import random

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from amf.common import crypto
from amf.common.index import id_to_int

TYPES_KEY = b"amf/types"          # the slot under which the type table is kept


class _Node:
    __slots__ = ("key", "leaf", "prio", "left", "right", "size", "min", "max")

    def __init__(self, key: int, leaf: bytes):
        self.key, self.leaf = key, leaf
        self.prio = random.random()
        self.left = self.right = None
        self.size, self.min, self.max = 1, key, key


def _upd(n: _Node) -> _Node:
    n.size, n.min, n.max = 1, n.key, n.key
    if n.left:
        n.size += n.left.size
        n.min = n.left.min
    if n.right:
        n.size += n.right.size
        n.max = n.right.max
    return n


def _split(t, key):
    """(keys < key, keys >= key)"""
    if t is None:
        return None, None
    if t.key < key:
        a, b = _split(t.right, key)
        t.right = a
        return _upd(t), b
    a, b = _split(t.left, key)
    t.left = b
    return a, _upd(t)


def _merge(a, b):
    if a is None:
        return b
    if b is None:
        return a
    if a.prio > b.prio:
        a.right = _merge(a.right, b)
        return _upd(a)
    b.left = _merge(a, b.left)
    return _upd(b)


def _rot_right(t: _Node) -> _Node:
    l = t.left
    t.left = l.right
    _upd(t)
    l.right = t
    return l


def _rot_left(t: _Node) -> _Node:
    r = t.right
    t.right = r.left
    _upd(t)
    r.left = t
    return r


def _insert(t, node: _Node):
    if t is None:
        return node
    if node.key < t.key:
        t.left = _insert(t.left, node)
        if t.left.prio > t.prio:
            t = _rot_right(t)
    else:
        t.right = _insert(t.right, node)
        if t.right.prio > t.prio:
            t = _rot_left(t)
    return _upd(t)


def _find(t, key: int):
    while t is not None and t.key != key:
        t = t.left if key < t.key else t.right
    return t


def _build(items, lo: int, hi: int, depth: int):
    """Balanced treap over items[lo:hi], already sorted by key (bulk load)."""
    if lo >= hi:
        return None
    mid = (lo + hi) // 2
    key, leaf = items[mid]
    node = _Node(key, leaf)
    node.prio = 1.0 - depth * 0.01 + random.random() * 0.005   # parents beat children
    node.left = _build(items, lo, mid, depth + 1)
    node.right = _build(items, mid + 1, hi, depth + 1)
    return _upd(node)


def _rank(t, key: int) -> int:
    """How many keys are < key."""
    r = 0
    while t is not None:
        if t.key < key:
            r += (t.left.size if t.left else 0) + 1
            t = t.right
        else:
            t = t.left
    return r


def _select(t, k: int):
    """The node with the k-th smallest key (0-based)."""
    while t is not None:
        ls = t.left.size if t.left else 0
        if k < ls:
            t = t.left
        elif k == ls:
            return t
        else:
            k -= ls + 1
            t = t.right
    return None


def _nodes_in(t, lo: int, hi: int, out: list) -> None:
    if t is None or t.max < lo or t.min >= hi:
        return
    _nodes_in(t.left, lo, hi, out)
    if lo <= t.key < hi:
        out.append(t)
    _nodes_in(t.right, lo, hi, out)


class Catalog:
    def __init__(self):
        self._root = None
        self._by_name: dict[str, tuple[bytes, bytes]] = {}
        self._name_by_id: dict[bytes, str] = {}

    # -- names, identifiers, leaves -------------------------------------------

    def __len__(self) -> int:
        return self._root.size if self._root else 0

    def get(self, name: str):
        return self._by_name.get(name)

    def name_of(self, obj_id: bytes):
        return self._name_by_id.get(obj_id)

    def leaf_of(self, obj_id: bytes) -> bytes:
        return self._by_name[self._name_by_id[obj_id]][1]

    def put(self, name: str, obj_id: bytes, leaf: bytes) -> None:
        """Record an upload, or a rewrite under the same name (same id, new leaf)."""
        held = self._by_name.get(name)
        if held is not None and held[0] != obj_id:
            raise ValueError("a name keeps its identifier")
        key = id_to_int(obj_id)
        if held is not None:
            _find(self._root, key).leaf = leaf
        else:
            self._root = _insert(self._root, _Node(key, leaf))
        self._by_name[name] = (obj_id, leaf)
        self._name_by_id[obj_id] = name

    def remove(self, name: str) -> None:
        obj_id, _leaf = self._by_name.pop(name)
        del self._name_by_id[obj_id]
        key = id_to_int(obj_id)
        left, rest = _split(self._root, key)
        _gone, right = _split(rest, key + 1)
        self._root = _merge(left, right)

    @classmethod
    def bulk_load(cls, entries):
        """Build from an iterable of (name, obj_id, leaf) in O(n): for setup and benchmarks."""
        cat = cls()
        items = []
        for name, obj_id, leaf in entries:
            if name in cat._by_name:
                raise ValueError("duplicate name in bulk load")
            cat._by_name[name] = (obj_id, leaf)
            cat._name_by_id[obj_id] = name
            items.append((id_to_int(obj_id), leaf))
        items.sort()
        cat._root = _build(items, 0, len(items), 0)
        return cat

    # -- the owner's own books for a range of identifiers ---------------------

    def count(self, lo: int, hi: int) -> int:
        """How many identifiers lie in [lo, hi): two ranks, O(log n)."""
        return _rank(self._root, hi) - _rank(self._root, lo)

    def ids_in(self, lo: int, hi: int) -> list[bytes]:
        out = []
        _nodes_in(self._root, lo, hi, out)
        return [n.key.to_bytes(crypto.ID_LEN, "big") for n in out]

    def leaves_in(self, lo: int, hi: int) -> list[bytes]:
        """The leaf values in [lo, hi), in identifier order: what the owner hashes up to a root."""
        out = []
        _nodes_in(self._root, lo, hi, out)
        return [n.leaf for n in out]

    # -- positions inside a tree (leaves sit in identifier order) ---------------

    def position(self, obj_id: bytes, lo: int) -> int:
        """Position of obj_id among the identifiers at or after lo: the catalog identifiers before it."""
        return _rank(self._root, id_to_int(obj_id)) - _rank(self._root, lo)

    def id_at(self, lo: int, k: int) -> bytes:
        """The identifier at position k of the tree whose range starts at lo."""
        node = _select(self._root, _rank(self._root, lo) + k)
        if node is None:
            raise IndexError("no identifier at that position")
        return node.key.to_bytes(crypto.ID_LEN, "big")

    # -- the encrypted copy kept at every provider, entry by entry ---------------

    def entries(self):
        """(name, obj_id, leaf) for every object, for a full copy at setup or after recovery."""
        return [(name, i, l) for name, (i, l) in self._by_name.items()]

    @classmethod
    def from_copy(cls, mk: bytes, copy: dict):
        """Rebuild catalog and type table from a provider's copy {slot: (version, blob)}; needs MK."""
        cat, types = cls(), {}
        for slot, (version, blob) in copy.items():
            data = open_entry(mk, slot, version, blob)
            if slot == TYPES_KEY:
                types = data["types"]
            elif not data.get("deleted"):
                cat.put(data["name"], slot, bytes.fromhex(data["leaf"]))
        return cat, types


def _entry_key(mk: bytes) -> bytes:
    return crypto._hkdf(mk, b"amf/catalog-key")


def _seal(mk: bytes, slot: bytes, version: int, data: dict) -> bytes:
    iv = os.urandom(crypto.IV_LEN)
    aad = slot + version.to_bytes(8, "big")           # the slot and version cannot be swapped
    return iv + AESGCM(_entry_key(mk)).encrypt(iv, json.dumps(data).encode("utf-8"), aad)


def seal_entry(mk: bytes, name: str, obj_id: bytes, leaf: bytes, version: int) -> bytes:
    return _seal(mk, obj_id, version, {"name": name, "leaf": leaf.hex()})


def seal_tombstone(mk: bytes, obj_id: bytes, version: int) -> bytes:
    return _seal(mk, obj_id, version, {"deleted": True})


def seal_types(mk: bytes, type_table: dict, version: int) -> bytes:
    return _seal(mk, TYPES_KEY, version, {"types": type_table})


def open_entry(mk: bytes, slot: bytes, version: int, blob: bytes) -> dict:
    """Raises cryptography.exceptions.InvalidTag if the blob, slot or version were tampered with."""
    aad = slot + version.to_bytes(8, "big")
    plain = AESGCM(_entry_key(mk)).decrypt(blob[:crypto.IV_LEN], blob[crypto.IV_LEN:], aad)
    return json.loads(plain.decode("utf-8"))


def newest(*copies: dict) -> dict:
    """Merge copies from several providers: for every slot, the highest version wins."""
    out: dict = {}
    for copy in copies:
        for slot, (version, blob) in copy.items():
            if slot not in out or version > out[slot][0]:
                out[slot] = (version, blob)
    return out