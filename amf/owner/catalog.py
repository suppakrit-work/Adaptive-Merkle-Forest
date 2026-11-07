"""The owner's catalog -- the state of the Verification Manager (III-A 1 b, Table I).

The catalog maps file names to identifiers and leaf hashes and keeps the
identifiers in identifier order, so the owner can, for any range [lo, hi):

* count the identifiers in it                      (n of every announced tree)
* list their leaf values in identifier order       (to compute the root itself,
                                                    n - 1 hash evaluations)
* turn a position inside a tree into an identifier, and the identifier into a
  file name                                        (naming the object, Phase 6)

The structure is the B+ tree of Section III-B 5): keyed by identifier, each
leaf entry carrying the leaf hash and each inner node the number of
identifiers below it, with a table from file names to identifiers beside it.
Insert, delete, rank, select and range count cost O(log_b N) node visits, and
the leaf values of a range lie on consecutive leaves.

The encrypted copy at every provider is kept entry by entry: each change sends
only the new entry, encrypted on its own under a key derived from MK and
tagged with the version of D; on recovery the highest version of each entry
wins, among the entries that open under MK.  The type table travels the same way.
"""
import json
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from amf.common import crypto
from amf.common.bptree import BPlusTree
from amf.common.index import id_to_int

TYPES_KEY = b"amf/types"          # the slot under which the type table is kept


class Catalog:
    def __init__(self):
        self._tree = BPlusTree()                                   # id (as int) -> leaf hash
        self._by_name: dict[str, tuple[bytes, bytes]] = {}
        self._name_by_id: dict[bytes, str] = {}

    # -- names, identifiers, leaves -------------------------------------------

    def __len__(self) -> int:
        return len(self._tree)

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
        self._tree.insert(id_to_int(obj_id), leaf)
        self._by_name[name] = (obj_id, leaf)
        self._name_by_id[obj_id] = name

    def remove(self, name: str) -> None:
        obj_id, _leaf = self._by_name.pop(name)
        del self._name_by_id[obj_id]
        self._tree.delete(id_to_int(obj_id))

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
        cat._tree = BPlusTree.bulk_load(items)
        return cat

    # -- the owner's own books for a range of identifiers ---------------------

    def count(self, lo: int, hi: int) -> int:
        """How many identifiers lie in [lo, hi): two ranks, O(log_b N)."""
        return self._tree.count(lo, hi)

    def ids_in(self, lo: int, hi: int) -> list[bytes]:
        return [k.to_bytes(crypto.ID_LEN, "big") for k, _ in self._tree.items(lo, hi)]

    def leaves_in(self, lo: int, hi: int) -> list[bytes]:
        """The leaf values in [lo, hi), in identifier order: what the owner hashes up to a root,
        read off consecutive leaves of the B+ tree."""
        return [leaf for _, leaf in self._tree.items(lo, hi)]

    # -- positions inside a tree (leaves sit in identifier order) ---------------

    def position(self, obj_id: bytes, lo: int) -> int:
        """Position of obj_id among the identifiers at or after lo: the catalog identifiers before it."""
        return self._tree.rank(id_to_int(obj_id)) - self._tree.rank(lo)

    def id_at(self, lo: int, k: int) -> bytes:
        """The identifier at position k of the tree whose range starts at lo."""
        try:
            key, _ = self._tree.select(self._tree.rank(lo) + k)
        except IndexError:
            raise IndexError("no identifier at that position") from None
        return key.to_bytes(crypto.ID_LEN, "big")

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

    @classmethod
    def from_copies(cls, mk: bytes, copies) -> tuple:
        """Rebuild from the copies of several providers: for every slot, the highest version that opens
        under MK.  A blob that does not open (damaged, or planted by a provider) is passed over."""
        candidates: dict = {}
        for copy in copies:
            for slot, entry in (copy.items() if isinstance(copy, dict) else ()):
                if isinstance(slot, bytes) and isinstance(entry, (tuple, list)) and len(entry) == 2:
                    candidates.setdefault(slot, []).append(tuple(entry))
        good = {}
        for slot, entries in candidates.items():
            for version, blob in sorted(entries, key=lambda e: e[0], reverse=True):
                try:
                    open_entry(mk, slot, version, blob)
                except (InvalidTag, ValueError, TypeError, AttributeError, OverflowError):
                    continue
                good[slot] = (version, blob)
                break
        return cls.from_copy(mk, good)


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
