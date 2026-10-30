"""The Auditor and the data user -- Section III-A 2), Phase 3.

An auditor checks integrity on the owner's behalf and holds no secret key.  It
challenges a provider for the identifier, ciphertext and path at positions of
its choosing, hashes the ciphertext itself, and checks it against the leaf
value returned and that value's path against the signed root.  A ciphertext
that fails names the provider and the position; a path that fails hands the
tree to Phase 5.  What an audit covers is set by the question: one object or a
set of them (only the trees whose ranges cover them), one owner (its live rows
from the DHT nodes and its current D from the owner, checking that none is
missing, then a sample of positions in each tree), or the whole forest.

A data user, anyone the owner allows to retrieve an object, gets the
identifier and the current D from the owner, looks the tree up in the index,
which returns the row with its path to D, and checks what it receives the same
way.  A reader whose row is older than the version named fetches the newer row
before it judges the answer.
"""
import random
from dataclasses import dataclass, field

from amf.common import crypto
from amf.common.index import digest_tree, freshest, verify_digest, verify_row, verify_row_in_digest
from amf.common.merkle import verify_path


@dataclass
class AuditReport:
    provider: str
    tree_id: bytes
    checked: list = field(default_factory=list)          # positions that passed
    bad_ciphertext: list = field(default_factory=list)   # CT does not hash to the leaf value returned
    bad_path: list = field(default_factory=list)         # the stored path does not reach the signed root

    @property
    def ok(self) -> bool:
        return not self.bad_ciphertext and not self.bad_path

    @property
    def tree_level(self) -> bool:
        """A failing path says the tree's stored hashes are wrong: Phase 5 takes over."""
        return bool(self.bad_path)


@dataclass
class OwnerAudit:
    own: bytes
    digest_ok: bool                                      # the live rows recompute to the owner's D
    trees: list = field(default_factory=list)            # one AuditReport per live tree

    @property
    def ok(self) -> bool:
        return self.digest_ok and all(t.ok for t in self.trees)


class ReadRefused(Exception):
    def __init__(self, provider: str, tree_id: bytes, position: int, kind: str):
        super().__init__(f"{kind} at {provider} tree {tree_id.hex()[:8]} position {position}")
        self.provider, self.tree_id, self.position, self.kind = provider, tree_id, position, kind


def check_answer(row, i, obj_id, ct, stored, path) -> str:
    """'ok', 'ciphertext' (the bytes changed, the path is sound) or 'path' (stored hashes wrong)."""
    leaf = crypto.leaf_hash(obj_id, ct)
    if verify_path(row.root, row.count, i, leaf, path):
        return "ok"
    if leaf != stored and verify_path(row.root, row.count, i, stored, path):
        return "ciphertext"
    return "path"


class Auditor:
    def __init__(self, owner_pks, s: int = 8, rng: random.Random | None = None):
        self.keys = {crypto.owner_tag(pk): pk for pk in owner_pks}   # public keys only; nothing secret
        self.s = s
        self.rng = rng or random.Random()

    def _row(self, own, tree_id, providers):
        """Ask more than one node, keep the row with the highest version, check its signature."""
        row = freshest([p.dht.get(own, tree_id) for p in providers])
        if row is None or not verify_row(self.keys[own], row):
            raise KeyError("no signed row for this tree")
        return row

    def live_rows(self, own, providers):
        tids = {tid for p in providers for (o, tid) in p.dht.rows if o == own}
        rows = [self._row(own, tid, providers) for tid in tids]
        return [r for r in rows if not r.retired]

    def audit_tree(self, row, providers, positions=None) -> AuditReport:
        prov = {p.name: p for p in providers}[row.provider]
        if positions is None:
            positions = self.rng.sample(range(row.count), min(self.s, row.count)) if row.count else []
        rep = AuditReport(row.provider, row.tree_id)
        for i, obj_id, ct, stored, path in prov.challenge(row.own, row.tree_id, positions):
            verdict = check_answer(row, i, obj_id, ct, stored, path) if row.covers(int.from_bytes(obj_id, "big")) \
                else "path"
            {"ok": rep.checked, "ciphertext": rep.bad_ciphertext, "path": rep.bad_path}[verdict].append(i)
        return rep

    def audit_ids(self, ids, digest, providers) -> list:
        """Asked about one object or a set: only the trees whose ranges cover them, each id read."""
        reader = Reader(list(self.keys.values()))
        reports = {}
        for obj_id in ids:
            row = reader.row_for(obj_id, digest, providers)
            rep = reports.setdefault(row.tree_id, self.audit_tree(row, providers))
            try:
                reader.read(obj_id, digest, providers)
            except ReadRefused as e:
                (rep.bad_ciphertext if e.kind == "ciphertext" else rep.bad_path).append(e.position)
        return list(reports.values())

    def audit_owner(self, digest, providers) -> OwnerAudit:
        """Asked about one owner: its live rows from the DHT nodes and its current D from the owner."""
        own = digest.own
        if not verify_digest(self.keys[own], digest):
            raise ValueError("D is not signed by this owner")
        rows = self.live_rows(own, providers)
        tree, _ = digest_tree(rows)
        result = OwnerAudit(own, tree.root == digest.root and tree.n == digest.count)
        for row in sorted(rows, key=lambda r: r.lo):
            result.trees.append(self.audit_tree(row, providers))
        return result

    def audit_all(self, digests, providers) -> list:
        """Asked about the whole forest: the same for every owner."""
        return [self.audit_owner(d, providers) for d in digests]


class Reader:
    """A data user: holds public keys only; gets the identifier and the current D from the owner."""

    def __init__(self, owner_pks):
        self.keys = {crypto.owner_tag(pk): pk for pk in owner_pks}

    def row_for(self, obj_id, digest, providers):
        own = crypto.owner_of(obj_id)
        if not verify_digest(self.keys[own], digest):
            raise ValueError("D is not signed by this owner")
        stale = None
        for p in providers:                               # ask more than one node if it must
            hit = p.dht.lookup(obj_id)
            if hit is None:
                continue
            row, i, _t, path = p.dht.row_with_path(own, hit.tree_id)
            if verify_row(self.keys[own], row) and verify_row_in_digest(digest, row, i, path):
                return row
            stale = row
        raise KeyError("no node holds a row that reaches the current D" + (" (stale rows only)" if stale else ""))

    def read(self, obj_id, digest, providers) -> bytes:
        """Returns the ciphertext once it reaches the signed root; decryption keys are outside this work."""
        row = self.row_for(obj_id, digest, providers)
        prov = {p.name: p for p in providers}[row.provider]
        tid, version, i, stored, ct, path = prov.read(obj_id)
        if tid == row.tree_id and version > row.version:
            # a read that crosses a write blames no one: the reader needs the newer row and D first
            raise ReadRefused(row.provider, row.tree_id, i, "newer version: refresh D")
        if tid != row.tree_id or version != row.version:
            raise ReadRefused(row.provider, row.tree_id, i, "wrong tree or version")
        verdict = check_answer(row, i, obj_id, ct, stored, path)
        if verdict != "ok":
            raise ReadRefused(row.provider, row.tree_id, i, verdict)
        return ct