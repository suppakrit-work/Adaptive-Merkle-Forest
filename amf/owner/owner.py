"""The Data Owner -- Section III-A 1), Phases 1, 2, 3 and 4, Algorithms 1 and 3.

Three managers, all on the owner's machine:

  Key/Signing Manager   holds MK and the signing key, whose public half everyone
                        else knows, and signs a new index entry once the
                        Verification Manager has accepted the announced root
                        behind it.
  Verification Manager  checks every read path, announced root and challenge
                        answer, and keeps the state this takes: the owner's copy
                        of the index, the catalog, the type table.
  Policy Manager        keeps the access counts and decides when a tree is split
                        or merged (policy.py); this file keeps the writes the
                        owner made, per half of each tree.

"No root gets into the index without passing Algorithm 3": the owner checks
every announced tree against its own order and its own catalog, and computes
the root itself (n - 1 hashes), or with one hash when a tree is split at its
median or two trees of nearly equal size are merged (Lemma 1).
"""
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from amf.common import crypto, waits
from amf.owner.catalog import TYPES_KEY, Catalog, seal_entry, seal_tombstone, seal_types
from amf.common.index import (ForestDigest, IndexRow, digest_tree, freshest_signed, generate_signing_key,
                       id_to_int, new_tree_id, retire, row_leaf, sign_digest, sign_row, tag_range,
                       verify_digest)
from amf.common.merkle import EMPTY_ROOT, left_size, node_hash, root_of, verify_path
from amf.common.orders import Order, sign_order


class VerificationError(Exception):
    """An announced root, a path or a challenge answer the Verification Manager did not accept."""


class ReadFault(VerificationError):
    """A read that failed: names the provider, the tree and the position, and which part was wrong."""

    def __init__(self, provider: str, tree_id: bytes, position: int, kind: str):
        super().__init__(f"{kind} at {provider} tree {tree_id.hex()[:8]} position {position}")
        self.provider, self.tree_id, self.position, self.kind = provider, tree_id, position, kind


class Owner:
    def __init__(self, mk: bytes | None = None, sk=None, s: int = 8, rng: random.Random | None = None,
                 parallel: bool = False):
        # Key/Signing Manager
        self.mk = mk if mk is not None else os.urandom(32)
        self.k_id = crypto.derive_id_key(self.mk)
        self.sk = sk if sk is not None else generate_signing_key()
        self.pk = self.sk.public_key()
        self.own = crypto.owner_tag(self.pk)
        # Verification Manager
        self.catalog = Catalog()
        self.index: dict[bytes, IndexRow] = {}          # the owner's copy: latest row per tree id
        self.digest: ForestDigest | None = None
        self.type_table: dict[str, int] = {}
        self.s = s                                      # positions sampled by a migration challenge
        self.rng = rng or random.SystemRandom()          # positions a provider cannot predict
        self.hashes = 0                                 # hash evaluations spent on checks (for the bench)
        self.digest_hashes = 0                          # hash evaluations spent keeping D (for the bench)
        self._dtree = None                              # D's tree over the live rows, and their positions
        self._dpos: dict[bytes, int] = {}
        # Policy Manager: writes the owner made, per half of each tree
        self.write_counts: dict[bytes, list] = {}
        self.policy = None                              # policy.PolicyManager attaches itself here
        # the owner keeps CT until the provider acknowledges
        self.unacked: dict[bytes, bytes] = {}
        self.providers: dict = {}
        # send to every DHT node at once (III-C, Parallelism); worth it when the providers are remote
        self.parallel = parallel
        self._pool = None
        self._lock = threading.Lock()                   # checks and signing one at a time (Parallelism)

    # -- Phase 1 ------------------------------------------------------------------

    def register_provider(self, provider) -> None:
        """pk is registered with every provider; a late provider still gets the whole table."""
        provider.register_owner(self.pk)
        self.providers[provider.name] = provider
        if self.index:
            provider.receive_batch(list(self.index.values()), self.digest)
        provider.store_catalog_entries(self.own, self._full_copy())

    def add_type(self, file_type: str, type_tag: int, provider_name: str) -> IndexRow:
        """One row per type tag, inside the owner's identifier space, with an empty tree."""
        if file_type in self.type_table or type_tag in self.type_table.values():
            raise ValueError("type or tag already in use")
        self.type_table[file_type] = type_tag
        lo, hi = tag_range(self.own, type_tag)
        row = IndexRow(self.own, new_tree_id(), lo, hi, provider_name, EMPTY_ROOT, 0, 1)
        return self._publish([row], {TYPES_KEY: None})[0]

    # -- the index: the owner is the only writer ---------------------------------

    def _publish(self, rows, changed=None) -> list:
        """Sign the rows and the new D, send them to every DHT node as one batch, then send the
        changed catalog entries, each encrypted on its own and tagged with D's version."""
        signed = [sign_row(self.sk, r) for r in rows]
        in_place = self._dtree is not None and all(r.tree_id in self._dpos and not r.retired for r in signed)
        for r in signed:
            self.index[r.tree_id] = r
        if in_place:                                    # a row changed in place: ceil(log2 T) hashes
            for r in signed:
                self.digest_hashes += 1 + self._dtree.update(self._dpos[r.tree_id], row_leaf(r))
        else:                                           # rows added or retired: D's tree is rebuilt
            self._dtree, ordered = digest_tree(self.index.values())
            self._dpos = {r.tree_id: i for i, r in enumerate(ordered)}
            self.digest_hashes += max(2 * len(ordered) - 1, 0)
        tree = self._dtree
        version = (self.digest.version if self.digest else 0) + 1
        self.digest = sign_digest(self.sk, ForestDigest(self.own, tree.root, tree.n, version))
        entries = {slot: (version, self._seal(slot, version)) for slot in changed} if changed else None

        def send(prov):
            prov.receive_batch(signed, self.digest)
            if entries:
                prov.store_catalog_entries(self.own, entries)
        self._to_every_provider(send)
        return signed

    def _to_every_provider(self, send) -> None:
        """Send the same message to every provider and wait for every answer.  With parallel=True the
        messages go out at once; each node still sees this owner's versions in order, since the owner
        waits for all of them before its next operation and a node keeps only a higher version."""
        provs = list(self.providers.values())
        if not self.parallel or len(provs) < 2:
            for p in provs:
                send(p)
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="amf-owner")
        with waits.waiting():
            for f in [self._pool.submit(send, p) for p in provs]:
                f.result()

    def _seal(self, slot: bytes, version: int) -> bytes:
        if slot == TYPES_KEY:
            return seal_types(self.mk, self.type_table, version)
        name = self.catalog.name_of(slot)
        if name is None:
            return seal_tombstone(self.mk, slot, version)
        return seal_entry(self.mk, name, slot, self.catalog.leaf_of(slot), version)

    def _full_copy(self) -> dict:
        v = self.digest.version if self.digest else 0
        out = {TYPES_KEY: (v, seal_types(self.mk, self.type_table, v))}
        for name, obj_id, leaf in self.catalog.entries():
            out[obj_id] = (v, seal_entry(self.mk, name, obj_id, leaf, v))
        return out

    def live_rows(self) -> list[IndexRow]:
        return [r for r in self.index.values() if not r.retired]

    def row_for(self, obj_id: bytes) -> IndexRow:
        v = id_to_int(obj_id)
        for r in self.live_rows():
            if r.covers(v):
                return r
        raise KeyError("no live tree covers this identifier")

    def identifier(self, name: str, file_type: str) -> bytes:
        return crypto.make_id(self.k_id, self.own, self.type_table[file_type], name)

    # -- Phase 2: Object Commitment ------------------------------------------------

    def prepare(self, name: str, data: bytes, file_type: str):
        obj_id = self.identifier(name, file_type)
        ct = crypto.encrypt(self.mk, obj_id, data)
        return obj_id, ct, crypto.leaf_hash(obj_id, ct)

    def _expected_leaves(self, row: IndexRow, obj_id: bytes, leaf: bytes | None) -> list:
        """The catalog's leaf values in the row's range with the pending change in place."""
        leaves = self.catalog.leaves_in(row.lo, row.hi)
        j = self.catalog.position(obj_id, row.lo)
        present = self.catalog.name_of(obj_id) is not None
        if leaf is None:
            return leaves[:j] + leaves[j + 1:]
        if present:
            return leaves[:j] + [leaf] + leaves[j + 1:]
        return leaves[:j] + [leaf] + leaves[j:]

    def _expected_leaves_many(self, row: IndexRow, pending: dict) -> list:
        """The same with several pending (id -> leaf) in place: a batch write."""
        merged = dict(zip(self.catalog.ids_in(row.lo, row.hi), self.catalog.leaves_in(row.lo, row.hi)))
        merged.update(pending)
        return [merged[i] for i in sorted(merged)]

    def _check_root(self, announced_root: bytes, leaves: list) -> None:
        """Algorithm 3, line 12: the root over the catalog's leaf values, n - 1 hashes."""
        self.hashes += max(len(leaves) - 1, 0)
        if root_of(leaves) != announced_root:
            raise VerificationError("the announced root is not the root over the owner's own leaves")

    def _check_write(self, row, announced, obj_id, leaf) -> None:
        tid, lo, hi, root, n = announced
        if tid != row.tree_id or (lo, hi) != (row.lo, row.hi):
            raise VerificationError("the provider announced another tree or range")
        expected = self._expected_leaves(row, obj_id, leaf)
        if n != len(expected):
            raise VerificationError(f"count {n}, the owner's books say {len(expected)}")
        self._check_root(root, expected)

    def upload(self, name: str, data: bytes, file_type: str) -> IndexRow:
        obj_id, ct, leaf = self.prepare(name, data, file_type)
        held = self.catalog.get(name)
        if held is not None and held[0] != obj_id:
            raise ValueError("that name belongs to another type")
        row = self.row_for(obj_id)
        if held is None and self.policy is not None:
            row = self.policy.before_write(row, obj_id)         # the size test, on the count a write would produce
        prov = self.providers[row.provider]
        self.unacked[obj_id] = ct
        announced = prov.write(obj_id, ct)
        try:
            self._check_write(row, announced, obj_id, leaf)
        except Exception:
            prov.discard_pending(self.own, row.tree_id)
            self.unacked.pop(obj_id, None)
            raise
        self.catalog.put(name, obj_id, leaf)
        new = self._publish([replace(row, root=announced[3], count=announced[4], version=row.version + 1)],
                            {obj_id: None})[0]
        if prov.acknowledged(self.own, new.tree_id, new.root):
            self.unacked.pop(obj_id, None)
        self._count_write(new, obj_id)
        return new

    def upload_many(self, files) -> list:
        """Phase 2 for many files at once: the objects bound for one tree go to the provider as one
        batch write, checked with all the pending leaves in place (Algorithm 3, line 12).  A tree is
        never taken past N_up: when a batch would, the tree is filled, split at its median, and the
        rest goes on into the halves.  Returns the rows signed, in order."""
        prepared = {}
        for name, data, file_type in files:
            obj_id, ct, leaf = self.prepare(name, data, file_type)
            held = self.catalog.get(name)
            if held is not None and held[0] != obj_id:
                raise ValueError("that name belongs to another type")
            if obj_id in prepared:
                raise ValueError(f"{name} twice in one batch")
            prepared[obj_id] = (name, ct, leaf)
        todo, signed = sorted(prepared), []
        while todo:
            row = self.row_for(todo[0])
            k = 0
            while k < len(todo) and row.covers(id_to_int(todo[k])):
                k += 1
            room = len(todo) if self.policy is None else max(0, self.policy.upper - row.count)
            new_here = sum(self.catalog.name_of(i) is None for i in todo[:k])
            if room < new_here and row.count >= 2:            # the tree would pass N_up: split it first
                self.policy.before_write(row, todo[0], k=new_here)
                continue
            chunk, fresh = [], 0                              # else fill the tree up to N_up and go on
            for i in todo[:k]:
                fresh += self.catalog.name_of(i) is None
                if fresh > room:
                    break
                chunk.append(i)
            k = len(chunk)
            prov = self.providers[row.provider]
            batch = [(i, prepared[i][1]) for i in chunk]
            for i, ct in batch:
                self.unacked[i] = ct
            announced = prov.write_many(batch)
            try:
                tid, lo, hi, root, n = announced
                if tid != row.tree_id or (lo, hi) != (row.lo, row.hi):
                    raise VerificationError("the provider announced another tree or range")
                expected = self._expected_leaves_many(row, {i: prepared[i][2] for i in chunk})
                if n != len(expected):
                    raise VerificationError(f"count {n}, the owner's books say {len(expected)}")
                self._check_root(root, expected)
            except Exception:
                prov.discard_pending(self.own, row.tree_id)
                for i in chunk:
                    self.unacked.pop(i, None)
                raise
            for i in chunk:
                self.catalog.put(prepared[i][0], i, prepared[i][2])
            new = self._publish([replace(row, root=root, count=n, version=row.version + 1)],
                                {i: None for i in chunk})[0]
            if prov.acknowledged(self.own, new.tree_id, new.root):
                for i in chunk:
                    self.unacked.pop(i, None)
            for i in chunk:
                self._count_write(new, i)
            signed.append(new)
            todo = todo[k:]
        return signed

    def delete(self, name: str) -> IndexRow:
        obj_id, _leaf = self.catalog.get(name)
        row = self.row_for(obj_id)
        prov = self.providers[row.provider]
        announced = prov.delete(obj_id)
        try:
            self._check_write(row, announced, obj_id, None)
        except Exception:
            prov.discard_pending(self.own, row.tree_id)
            raise
        self._count_write(row, obj_id)
        self.catalog.remove(name)
        return self._publish([replace(row, root=announced[3], count=announced[4], version=row.version + 1)],
                             {obj_id: None})[0]

    def _count_write(self, row: IndexRow, obj_id: bytes) -> None:
        if row.count == 0:
            return
        j = self.catalog.position(obj_id, row.lo)
        halves = self.write_counts.setdefault(row.tree_id, [0, 0])
        halves[0 if j < left_size(row.count) else 1] += 1

    # -- Phase 3: the owner's own reads --------------------------------------------

    def read(self, name: str) -> bytes:
        obj_id, _leaf = self.catalog.get(name)
        row = self.row_for(obj_id)
        prov = self.providers[row.provider]
        tid, version, i, stored, ct, path = prov.read(obj_id)
        if tid != row.tree_id or version != row.version:
            raise ReadFault(row.provider, row.tree_id, i, "wrong tree or version")
        if i != self.catalog.position(obj_id, row.lo):
            raise ReadFault(row.provider, row.tree_id, i, "wrong position")
        if ct is None:
            raise ReadFault(row.provider, row.tree_id, i, "missing")
        leaf = crypto.leaf_hash(obj_id, ct)
        if not verify_path(row.root, row.count, i, leaf, path):
            kind = "ciphertext" if verify_path(row.root, row.count, i, stored, path) else "path"
            raise ReadFault(row.provider, row.tree_id, i, kind)
        return crypto.decrypt(self.mk, obj_id, ct)

    def reader_view(self, name: str, file_type: str):
        """What a reader other than the owner gets from the owner: the identifier and the current D."""
        return self.identifier(name, file_type), self.digest

    # -- Phase 4: restructuring, checked by Algorithm 3 -------------------------------

    def _order(self, kind, trees, new_ids=(), boundary=0, source="", destination="", version=0) -> Order:
        return sign_order(self.sk, Order(kind, self.own, tuple(trees), tuple(new_ids), boundary,
                                         source, destination, version))

    def _check_announced_range_count(self, ann, lo: int, hi: int) -> None:
        """Algorithm 3, lines 1-3: the range the order implies, the count of catalog identifiers in it."""
        _tid, a_lo, a_hi, _root, n = ann
        if (a_lo, a_hi) != (lo, hi):
            raise VerificationError("the announced range is not the one the order implies")
        if n != self.catalog.count(lo, hi):
            raise VerificationError(f"count {n}, the owner's books say {self.catalog.count(lo, hi)}")

    def median_boundary(self, row: IndexRow) -> int:
        """p = the identifier at position ceil(n/2): the two new trees are the children of the old root."""
        if row.count < 2:
            raise ValueError("nothing to split")
        return id_to_int(self.catalog.id_at(row.lo, left_size(row.count)))

    def split(self, tree_id: bytes, p: int | None = None):
        """Decomposition at boundary p (the median by default). Returns the two new rows."""
        row = self.index[tree_id]
        prov = self.providers[row.provider]
        p = self.median_boundary(row) if p is None else p
        if not row.lo < p < row.hi:
            raise ValueError("the boundary must lie inside the range")
        id_a, id_b = new_tree_id(), new_tree_id()
        order = self._order("split", [tree_id], [id_a, id_b], p, row.provider, version=row.version)
        ann_a, ann_b = prov.decompose(order)
        with self._lock:
            try:
                if (ann_a[0], ann_b[0]) != (id_a, id_b):
                    raise VerificationError("the provider announced other tree identifiers")
                self._check_announced_range_count(ann_a, row.lo, p)
                self._check_announced_range_count(ann_b, p, row.hi)
                r_a, r_b = ann_a[3], ann_b[3]
                if row.count >= 2 and ann_b[4] >= 1 and p == self.median_boundary(row):
                    self.hashes += 1                                       # Algorithm 3, line 6
                    if node_hash(r_a, r_b) != row.root:
                        raise VerificationError("H(r_a || r_b) is not the signed root")
                else:
                    self._check_root(r_a, self.catalog.leaves_in(row.lo, p))
                    self._check_root(r_b, self.catalog.leaves_in(p, row.hi))
            except Exception:
                prov.discard_pending(self.own, id_a, id_b)
                raise
            a = IndexRow(self.own, id_a, row.lo, p, row.provider, r_a, ann_a[4], 1)
            b = IndexRow(self.own, id_b, p, row.hi, row.provider, r_b, ann_b[4], 1)
            self._forget_counts(tree_id)
            return self._publish([a, b, retire(row)])

    def merge(self, id_a: bytes, id_b: bytes) -> IndexRow:
        """Composition of two adjacent trees at the same provider. Returns the new row."""
        a, b = self.index[id_a], self.index[id_b]
        if a.provider != b.provider or a.hi != b.lo:
            raise ValueError("merge needs adjacent ranges at the same provider")
        prov = self.providers[a.provider]
        id_ab = new_tree_id()
        order = self._order("merge", [id_a, id_b], [id_ab], source=a.provider, version=a.version)
        (ann,) = prov.compose(order)
        with self._lock:
            try:
                if ann[0] != id_ab:
                    raise VerificationError("the provider announced another tree identifier")
                self._check_announced_range_count(ann, a.lo, b.hi)
                r_new, n_new = ann[3], ann[4]
                if b.count >= 1 and a.count == left_size(n_new):
                    self.hashes += 1                                       # Algorithm 3, line 7
                    if node_hash(a.root, b.root) != r_new:
                        raise VerificationError("H(r_a || r_b) is not the announced root")
                else:
                    self._check_root(r_new, self.catalog.leaves_in(a.lo, b.hi))
            except Exception:
                prov.discard_pending(self.own, id_ab)
                raise
            self._forget_counts(id_a, id_b)
            new = IndexRow(self.own, id_ab, a.lo, b.hi, a.provider, r_new, n_new, 1)
            return self._publish([new, retire(a), retire(b)])[0]

    def migrate(self, tree_id: bytes, destination: str) -> IndexRow:
        """Move a whole tree through the Migrate Pipe; the source keeps its copy until released."""
        row = self.index[tree_id]
        if destination == row.provider:
            raise ValueError("the tree is already there")
        src, dst = self.providers[row.provider], self.providers[destination]
        order = self._order("migrate", [tree_id], source=row.provider, destination=destination,
                            version=row.version)
        (ann,) = dst.import_from(order, src)
        try:
            if ann[0] != tree_id:
                raise VerificationError("the destination announced another tree")
            self._check_announced_range_count(ann, row.lo, row.hi)
            if ann[3] != row.root:                                     # Algorithm 3, line 8
                raise VerificationError("a migration changes nothing inside the tree")
            self._challenge_pending(dst, row)                          # Algorithm 3, line 9
        except Exception:
            dst.discard_pending(self.own, tree_id)
            raise
        return self._publish([replace(row, provider=destination, version=row.version + 1)])[0]

    def _challenge_pending(self, dst, row: IndexRow) -> None:
        if row.count == 0:
            return
        sample = self.rng.sample(range(row.count), min(self.s, row.count))
        answers = {a[0]: a for a in dst.pending_challenge(self.own, row.tree_id, sample)}
        for i in sample:                                   # an answer for every position asked, at that position
            if i not in answers:
                raise VerificationError(f"no answer for position {i} at the destination")
            _i, obj_id, ct, _stored, path = answers[i]
            if obj_id != self.catalog.id_at(row.lo, i):
                raise VerificationError(f"position {i} holds another identifier")
            if ct is None:
                raise VerificationError(f"position {i} is missing at the destination")
            leaf = crypto.leaf_hash(obj_id, ct)
            self.hashes += 1 + len(path)
            if leaf != self.catalog.leaf_of(obj_id) or not verify_path(row.root, row.count, i, leaf, path):
                raise VerificationError(f"challenge at position {i} failed at the destination")

    def release(self, tree_id: bytes, source: str) -> int:
        """The second signed order: the migration source drops its copy."""
        row = self.index[tree_id]
        order = self._order("release", [tree_id], source=source, version=row.version)
        return self.providers[source].release(order)

    def evacuate(self, provider_name: str, trees=None, repairs=None) -> list:
        """FailedCheck of Algorithm 2, after a failed check of Phase 3, 5 or 6: repair the named objects through Phase 2,
        then move the trees that failed (all of that provider's trees when trees is None) to the
        providers holding the fewest of the owner's trees.  repairs: {name: (data, type) or None}."""
        for name, fix in (repairs or {}).items():
            if fix is None:
                self.delete(name)
            else:
                self.upload(name, fix[0], fix[1])
        if trees is None:
            trees = [r.tree_id for r in self.live_rows() if r.provider == provider_name]
        return [self._move_away(tid, provider_name) for tid in trees]

    def _move_away(self, tree_id: bytes, provider_name: str) -> IndexRow:
        """One tree to the provider holding the fewest of the owner's trees.  If the migration fails the
        row still names the source: the owner recounts the source and tries another destination (IV-D)."""
        tried = {provider_name}
        while True:
            held = {n: 0 for n in self.providers if n not in tried}
            if not held:
                if len(tried) == 1:
                    raise ValueError("no other provider to move to")
                raise VerificationError(f"no destination accepted tree {tree_id.hex()[:8]}")
            for r in self.live_rows():
                if r.provider in held:
                    held[r.provider] += 1
            target = min(sorted(held), key=lambda n: held[n])
            try:
                return self.migrate(tree_id, target)
            except Exception as e:
                tried.add(target)
                row = self.index[tree_id]
                if self.providers[row.provider].recomputed_root(self.own, tree_id) != row.root:
                    raise VerificationError(f"{row.provider} no longer recomputes the signed root of tree "
                                            f"{tree_id.hex()[:8]}: localize and repair it first") from e

    def _forget_counts(self, *tree_ids: bytes) -> None:
        """The write counts of trees that a split or a merge retires; the new trees count afresh."""
        for t in tree_ids:
            self.write_counts.pop(t, None)

    # -- rebuilding the owner's state from the providers (III-A 1 b) -------------------

    @classmethod
    def recover(cls, mk: bytes, sk, providers, s: int = 8):
        """Whoever holds MK and the signing key rebuilds catalog, type table, index and D.  Only what
        verifies is used: the newest signed row of each tree, the newest signed D, the newest catalog
        entry that opens under MK; and the rows must recompute to that D."""
        owner = cls(mk=mk, sk=sk, s=s)
        copies = [p.catalog_copy.get(owner.own, {}) for p in providers]
        owner.catalog, owner.type_table = Catalog.from_copies(mk, copies)
        tids = {tid for p in providers for tid in p.dht.tree_ids(owner.own)}
        for tid in tids:
            row = freshest_signed(owner.pk, [p.dht.get(owner.own, tid) for p in providers], (owner.own, tid))
            if row is not None:
                owner.index[row.tree_id] = row
        digests = [p.dht.digest(owner.own) for p in providers]
        digests = [d for d in digests if d is not None and verify_digest(owner.pk, d)]
        owner.digest = max(digests, key=lambda d: d.version) if digests else None
        if owner.digest is not None:
            tree, _ = digest_tree(owner.index.values())
            if (tree.root, tree.n) != (owner.digest.root, owner.digest.count):
                raise VerificationError("the rows the providers hold do not recompute to the latest D")
        for p in providers:
            owner.providers[p.name] = p
        return owner
