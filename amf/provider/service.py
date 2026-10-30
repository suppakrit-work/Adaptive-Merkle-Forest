"""The provider's Storage Service -- Section III-A 3).

Every provider runs the same Storage Service.  It keeps the ciphertexts and the
trees it is responsible for, with H(id || CT) of each object as a leaf, and
runs three components: the Proof Engine answers with a read path, an announced
root or a challenge answer; the Shadow Area is where a new tree is built during
a write, a split, a merge or a migration, and a tree there is not part of the
forest until the owner signs its entry; the Migrate Pipe moves a whole tree to
or from another provider on a signed order from the owner, the source keeping
its copy until the owner releases it.  Beneath the service sit a DHT node (the
provider's copy of the Distributed Index, index.py) and a Merkle Tree Index,
which holds the nodes of each tree and maps each identifier to its tree and
position (TreeState).  A provider holds trees of many owners side by side;
trees are keyed by (own, tid).

Every change of a root goes the same way: the provider stores what it was
given, builds the changed or new tree in the Shadow Area and announces range,
root and count; the live trees keep serving until the signed rows arrive.  A
row for a pending tree promotes it (and that is the acknowledgement); a
retired row removes a live tree; a row that moves one of our trees to another
provider turns our copy into a retained one, kept until a release order.

An object the provider no longer has is answered with no ciphertext (None), and
its recomputed leaf is MISSING, so a read, a challenge and a recount all show it.
"""
from bisect import bisect_left
from dataclasses import dataclass

from amf.common import crypto
from amf.common.index import DHTNode, IndexRow
from amf.common.merkle import EMPTY_ROOT, MerkleTree, left_size
from amf.common.orders import Order, verify_order

MISSING = bytes(32)                   # the recomputed leaf of an object the provider does not hold


@dataclass
class TreeState:
    """One tree in the Merkle Tree Index: identifiers in order, their leaf values, the nodes."""
    own: bytes
    tree_id: bytes
    lo: int
    hi: int
    ids: list            # identifiers in identifier order
    leaves: list         # stored leaf value of each, same order
    tree: MerkleTree
    version: int = 0     # the version of the signed row this tree went live under

    @classmethod
    def build(cls, own, tree_id, lo, hi, ids, leaves):
        return cls(own, tree_id, lo, hi, list(ids), list(leaves), MerkleTree(list(leaves)))

    @property
    def key(self):
        return self.own, self.tree_id

    @property
    def root(self) -> bytes:
        return self.tree.root

    @property
    def count(self) -> int:
        return len(self.ids)

    def position(self, obj_id: bytes) -> int:
        i = bisect_left(self.ids, obj_id)
        if i < len(self.ids) and self.ids[i] == obj_id:
            return i
        raise KeyError("identifier not in this tree")

    def holds(self, obj_id: bytes) -> bool:
        i = bisect_left(self.ids, obj_id)
        return i < len(self.ids) and self.ids[i] == obj_id

    def announce(self):
        """What the provider announces for a pending tree: (tid, lo, hi, root, count)."""
        return self.tree_id, self.lo, self.hi, self.root, self.count


@dataclass
class Pending:
    """A tree in the Shadow Area.  kind 'write' (same tid, one object added, rewritten or
    deleted) or 'new' (a split half, a merge result or a migrated-in tree)."""
    state: TreeState
    kind: str
    staged: dict | None = None          # id -> CT stored with the write or the migration, None = deletion
    gone: bytes | None = None           # the identifier a deletion removes


class Provider:
    def __init__(self, name: str, owner_pks=(), objects=None):
        self.name = name
        self.objects = objects if objects is not None else {}      # id -> CT
        self.live: dict[tuple, TreeState] = {}
        self.shadow: dict[tuple, Pending] = {}
        self.retained: dict[tuple, TreeState] = {}                  # migrated away, kept until released
        self.dht = DHTNode(owner_pks)
        self.catalog_copy: dict[bytes, dict] = {}                   # own -> {slot: (version, blob)}
        self.reads: dict[bytes, dict] = {}                          # own -> {(own, tid): [left half, right half]}
        self._walk_cache: dict[tuple, list] = {}

    def register_owner(self, pk) -> bytes:
        """Phase 1: the owner's public key is registered with every provider."""
        return self.dht.register(pk)

    def _pk(self, own: bytes):
        return self.dht.keys.get(own)

    # -- the index copy, and what a new signed batch does here -------------------

    def receive_batch(self, rows, digest=None) -> bool:
        """Accept the rows of one operation (all or none) and act on what they say about our trees."""
        if not self.dht.accept_batch(rows, digest):
            return False
        for row in rows:
            self._apply(row)
        return True

    def receive_row(self, row: IndexRow) -> bool:
        return self.receive_batch([row])

    def _apply(self, row: IndexRow) -> None:
        key = row.key
        if row.retired:
            self.live.pop(key, None)                    # objects stay: they belong to the new trees
            return
        if row.provider != self.name:
            if key in self.live:                         # migrated away: keep the copy until released
                self.retained[key] = self.live.pop(key)
            return
        pend = self.shadow.get(key)
        live = self.live.get(key)
        if pend is not None and pend.state.root == row.root and pend.state.count == row.count:
            self._promote(key, pend, row.version)
        elif live is None and row.count == 0 and row.root == EMPTY_ROOT:
            self.live[key] = TreeState.build(row.own, row.tree_id, row.lo, row.hi, [], [])
            self.live[key].version = row.version
        elif live is not None and (live.root, live.count) == (row.root, row.count):
            live.version = row.version                   # the same tree under a newer row

    def _promote(self, key, pend: Pending, version: int) -> None:
        if pend.gone is not None:
            self.objects.pop(pend.gone, None)
        if pend.staged:
            self.objects.update(pend.staged)
        pend.state.version = version
        self.live[key] = pend.state
        del self.shadow[key]
        self.retained.pop(key, None)                    # a tree that came back: the new copy is the one in use
        self._walk_cache.pop(key, None)

    def acknowledged(self, own: bytes, tree_id: bytes, root: bytes) -> bool:
        """Has the tree with exactly this root gone live here?  The owner keeps CT until then."""
        t = self.live.get((own, tree_id))
        return t is not None and t.root == root

    def store_catalog_entries(self, own: bytes, entries: dict) -> None:
        """The encrypted catalog copy, entry by entry: keep the highest version of each slot."""
        copy = self.catalog_copy.setdefault(own, {})
        for slot, (version, blob) in entries.items():
            if slot not in copy or version > copy[slot][0]:
                copy[slot] = (version, blob)

    def discard_pending(self, own: bytes, *tree_ids: bytes) -> None:
        """A check that fails leaves a Shadow Area to be thrown away."""
        for t in tree_ids:
            self.shadow.pop((own, t), None)

    def tree_for(self, obj_id: bytes) -> TreeState:
        row = self.dht.lookup(obj_id)
        if row is None or row.key not in self.live:
            raise KeyError("no live tree of ours covers this identifier")
        return self.live[row.key]

    # -- writes: store, build in the Shadow Area, announce ------------------------

    def write(self, obj_id: bytes, ct: bytes):
        """Phase 2 step 3: store CT, build the new tree, announce (tid, lo, hi, root, count)."""
        t = self.tree_for(obj_id)
        if t.key in self.shadow:
            raise RuntimeError("an operation is already pending on this tree")
        leaf = crypto.leaf_hash(obj_id, ct)
        j = bisect_left(t.ids, obj_id)
        if j < len(t.ids) and t.ids[j] == obj_id:                       # rewrite under the same name
            ids, leaves = t.ids, t.leaves[:j] + [leaf] + t.leaves[j + 1:]
        else:
            ids, leaves = t.ids[:j] + [obj_id] + t.ids[j:], t.leaves[:j] + [leaf] + t.leaves[j:]
        new = TreeState.build(t.own, t.tree_id, t.lo, t.hi, ids, leaves)
        self.shadow[t.key] = Pending(new, "write", staged={obj_id: ct})
        return new.announce()

    def delete(self, obj_id: bytes):
        t = self.tree_for(obj_id)
        if t.key in self.shadow:
            raise RuntimeError("an operation is already pending on this tree")
        j = t.position(obj_id)
        new = TreeState.build(t.own, t.tree_id, t.lo, t.hi, t.ids[:j] + t.ids[j + 1:],
                              t.leaves[:j] + t.leaves[j + 1:])
        self.shadow[t.key] = Pending(new, "write", gone=obj_id)
        return new.announce()

    # -- restructuring (Phase 4): decomposition, composition, the Migrate Pipe ----

    def _order_ok(self, order: Order, kind: str, role: str) -> bool:
        if order.kind != kind or not verify_order(self._pk(order.own), order):
            return False
        return (order.destination if role == "destination" else order.source) == self.name

    def _held(self, own: bytes, tree_id: bytes, version: int | None = None) -> TreeState:
        """A live tree of ours whose current row has the version the order names (if it names one)."""
        row = self.dht.get(own, tree_id)
        t = self.live.get((own, tree_id))
        if row is None or row.retired or row.provider != self.name or t is None:
            raise KeyError("no live tree of ours with that identifier")
        if version is not None and row.version != version:
            raise PermissionError("the order is not for the current row of the tree")
        return t

    def decompose(self, order: Order):
        """Split a live tree at boundary p into two pending trees; announce both."""
        if not self._order_ok(order, "split", "source"):
            raise PermissionError("no valid split order from the owner")
        (tid,), (id_a, id_b), p = order.trees, order.new_ids, order.boundary
        t = self._held(order.own, tid, order.version)
        if not t.lo < p < t.hi:
            raise ValueError("the boundary must lie inside the range")
        k = bisect_left(t.ids, p.to_bytes(crypto.ID_LEN, "big"))
        a = TreeState.build(t.own, id_a, t.lo, p, t.ids[:k], t.leaves[:k])
        b = TreeState.build(t.own, id_b, p, t.hi, t.ids[k:], t.leaves[k:])
        self.shadow[a.key] = Pending(a, "new")
        self.shadow[b.key] = Pending(b, "new")
        return [a.announce(), b.announce()]

    def compose(self, order: Order):
        """Merge two live trees with adjacent ranges into one pending tree; announce it."""
        if not self._order_ok(order, "merge", "source"):
            raise PermissionError("no valid merge order from the owner")
        (id_a, id_b), (id_ab,) = order.trees, order.new_ids
        a, b = self._held(order.own, id_a, order.version), self._held(order.own, id_b)
        if a.hi != b.lo:
            raise ValueError("the ranges are not adjacent")
        ab = TreeState.build(a.own, id_ab, a.lo, b.hi, a.ids + b.ids, a.leaves + b.leaves)
        self.shadow[ab.key] = Pending(ab, "new")
        return [ab.announce()]

    def export_tree(self, order: Order):
        """The Migrate Pipe, source side: hand a tree over only on the owner's signed order, and
        only for the row version it names (an old order does not open the pipe again)."""
        if not self._order_ok(order, "migrate", "source"):
            raise PermissionError("no valid migration order from the owner")
        row = self.dht.get(order.own, order.trees[0])
        if row is None or row.retired or row.version != order.version or row.provider != self.name:
            raise PermissionError("the order is not for the current row of a tree held here")
        t = self.live[(order.own, order.trees[0])]
        missing = [i for i in t.ids if i not in self.objects]
        if missing:
            raise KeyError(f"{len(missing)} object(s) of the tree missing at the source")
        return list(t.ids), [self.objects[i] for i in t.ids]

    def check_import(self, order: Order) -> IndexRow:
        """The destination's own checks before it pulls anything: a signed order addressed to it,
        for the current row of the tree."""
        if not self._order_ok(order, "migrate", "destination"):
            raise PermissionError("no valid migration order from the owner")
        row = self.dht.get(order.own, order.trees[0])
        if row is None or row.retired or row.version != order.version:
            raise KeyError("no current signed row for the tree being moved")
        return row

    def import_from(self, order: Order, source: "Provider"):
        """The Migrate Pipe, destination side: pull the tree from the source, check what arrived
        against the signed root before announcing anything."""
        row = self.check_import(order)
        tid = order.trees[0]
        ids, cts = source.export_tree(order)
        state = TreeState.build(order.own, tid, row.lo, row.hi, ids,
                                [crypto.leaf_hash(i, c) for i, c in zip(ids, cts)])
        if state.root != row.root or state.count != row.count:
            raise ValueError("what arrived does not match the signed root")
        self.shadow[state.key] = Pending(state, "new", staged=dict(zip(ids, cts)))
        return [state.announce()]

    def pending_challenge(self, own: bytes, tree_id: bytes, positions):
        """Identifier, ciphertext, stored leaf and path at positions of a pending (migrated-in) tree."""
        pend = self.shadow[(own, tree_id)]
        st = pend.state
        return [(i, st.ids[i], pend.staged.get(st.ids[i]), st.leaves[i], st.tree.path(i)) for i in positions]

    def release(self, order: Order) -> int:
        """A migration source drops its copy on the owner's second signed order. Returns objects dropped."""
        if not self._order_ok(order, "release", "source"):
            raise PermissionError("no valid release order from the owner")
        t = self.retained.pop((order.own, order.trees[0]), None)
        if t is None:
            return 0
        dropped = 0
        for obj_id in t.ids:
            if not self._in_use(obj_id):                # the tree, or part of it, may have come back
                dropped += self.objects.pop(obj_id, None) is not None
        return dropped

    def _in_use(self, obj_id: bytes) -> bool:
        row = self.dht.lookup(obj_id)
        t = self.live.get(row.key) if row is not None else None
        return t is not None and t.holds(obj_id)

    # -- the Proof Engine ------------------------------------------------------

    def _count_read(self, t: TreeState, i: int) -> None:
        halves = self.reads.setdefault(t.own, {}).setdefault(t.key, [0, 0])
        halves[0 if i < left_size(t.count) else 1] += 1

    def read(self, obj_id: bytes):
        """The read path: (tid, version of the row, position, stored leaf, CT or None, sibling hashes)."""
        t = self.tree_for(obj_id)
        i = t.position(obj_id)
        self._count_read(t, i)
        return t.tree_id, t.version, i, t.leaves[i], self.objects.get(obj_id), t.tree.path(i)

    def report_reads(self, own: bytes) -> dict:
        """Reads served per half of each of this owner's trees since its last report, its own included."""
        return {k: tuple(v) for k, v in self.reads.pop(own, {}).items()}

    def challenge(self, own: bytes, tree_id: bytes, positions):
        """Identifier, ciphertext (or None), stored leaf and path at positions of the challenger's choosing."""
        t = self.live[(own, tree_id)]
        return [(i, t.ids[i], self.objects.get(t.ids[i]), t.leaves[i], t.tree.path(i)) for i in positions]

    # -- what fault localization asks for (Phase 5) --------------------------------

    def recomputed_leaves(self, own: bytes, tree_id: bytes) -> list:
        t = self.live[(own, tree_id)]
        cts = [self.objects.get(i) for i in t.ids]
        return [MISSING if ct is None else crypto.leaf_hash(i, ct) for i, ct in zip(t.ids, cts)]

    def recomputed_root(self, own: bytes, tree_id: bytes) -> bytes:
        """r': the root recomputed from the objects held right now (a recount)."""
        self._walk_cache[(own, tree_id)] = self.recomputed_leaves(own, tree_id)
        return MerkleTree(self._walk_cache[(own, tree_id)]).root

    def stored_leaves(self, own: bytes, tree_id: bytes) -> list:
        return list(self.live[(own, tree_id)].leaves)

    def stored_root(self, own: bytes, tree_id: bytes) -> bytes:
        return self.live[(own, tree_id)].root

    def recount(self, own: bytes) -> dict:
        """The question that opens Phase 5, for all of this owner's trees here in one answer:
        tid -> (stored root, r')."""
        out = {}
        for tid in self.dht.tree_ids(own):
            if (own, tid) in self.live:
                out[tid] = (self.stored_root(own, tid), self.recomputed_root(own, tid))
        return out

    def halves(self, own: bytes, tree_id: bytes, k: int, lo: int, hi: int):
        """For the two children of node k (leaves [lo, hi)), st and rec."""
        if hi - lo < 2:
            raise ValueError("a single leaf has no children")
        t = self.live[(own, tree_id)]
        rec = self._walk_cache.get((own, tree_id)) or self.recomputed_leaves(own, tree_id)
        mid = lo + left_size(hi - lo)
        return [(2 * k, lo, mid, t.tree.node(2 * k), MerkleTree(rec[lo:mid]).root),
                (2 * k + 1, mid, hi, t.tree.node(2 * k + 1), MerkleTree(rec[mid:hi]).root)]

    def round_halves(self, own: bytes, tree_id: bytes, nodes) -> list:
        """One round of the walk in one exchange: the halves of every node [(k, lo, hi), ...] asked."""
        return [self.halves(own, tree_id, k, lo, hi) for k, lo, hi in nodes]

    def report_mismatches(self, own: bytes, tree_id: bytes):
        """One exchange: the positions where stored and recomputed leaf values differ, each with
        its stored leaf and stored path."""
        t = self.live[(own, tree_id)]
        rec = self._walk_cache.get((own, tree_id)) or self.recomputed_leaves(own, tree_id)
        return [(i, t.leaves[i], t.tree.path(i)) for i in range(t.count) if t.leaves[i] != rec[i]]

    # -- fault injection, for demos and tests ------------------------------------------

    def corrupt(self, obj_id: bytes, at: int = 0) -> None:
        """Flip one byte of a stored ciphertext (a failing disk, a bug, an intruder)."""
        ct = bytearray(self.objects[obj_id])
        ct[at % len(ct)] ^= 0xFF
        self.objects[obj_id] = bytes(ct)

    def stats(self) -> dict:
        return {"objects": len(self.objects), "live": len(self.live), "shadow": len(self.shadow),
                "retained": len(self.retained), "rows": len(self.dht.rows)}
