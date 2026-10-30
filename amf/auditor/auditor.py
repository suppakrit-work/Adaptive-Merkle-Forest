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
way.  It asks every node it knows and takes the newest D signed by the owner
among its own and theirs, so between two contacts with the owner its view is
as fresh as one honest node makes it.  A reader whose row is older than the
version named fetches the newer row before it judges the answer.
"""
import random
from dataclasses import dataclass, field

from amf.common import crypto
from amf.common.index import digest_tree, freshest_signed, verify_digest, verify_row, verify_row_in_digest
from amf.common.merkle import verify_path


@dataclass
class AuditReport:
    provider: str
    tree_id: bytes
    checked: list = field(default_factory=list)          # positions that passed
    bad_ciphertext: list = field(default_factory=list)   # CT does not hash to the leaf value returned
    bad_path: list = field(default_factory=list)         # the stored path does not reach the signed root
    missing: list = field(default_factory=list)          # no ciphertext, or no answer for that position

    @property
    def ok(self) -> bool:
        return not self.bad_ciphertext and not self.bad_path and not self.missing

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
    """'ok', 'missing' (no ciphertext), 'ciphertext' (the bytes changed, the path is sound) or 'path'
    (stored hashes wrong)."""
    if ct is None:
        return "missing"
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
        self.rng = rng or random.SystemRandom()

    def _row(self, own, tree_id, providers):
        """Ask more than one node and keep the signed row with the highest version (None if none is signed)."""
        return freshest_signed(self.keys[own], [p.dht.get(own, tree_id) for p in providers], (own, tree_id))

    def live_rows(self, own, providers):
        tids = {tid for p in providers for tid in p.dht.tree_ids(own)}
        rows = [self._row(own, tid, providers) for tid in tids]
        return [r for r in rows if r is not None and not r.retired]

    def audit_tree(self, row, providers, positions=None) -> AuditReport:
        prov = {p.name: p for p in providers}[row.provider]
        if positions is None:
            positions = self.rng.sample(range(row.count), min(self.s, row.count)) if row.count else []
        positions = list(positions)
        rep = AuditReport(row.provider, row.tree_id)
        answers = {}
        for a in prov.challenge(row.own, row.tree_id, positions):
            answers.setdefault(a[0], a)
        for i in positions:                               # an answer for every position asked, at that position
            if i not in answers:
                rep.missing.append(i)
                continue
            _i, obj_id, ct, stored, path = answers[i]
            verdict = check_answer(row, i, obj_id, ct, stored, path) if row.covers(int.from_bytes(obj_id, "big")) \
                else "path"
            {"ok": rep.checked, "ciphertext": rep.bad_ciphertext, "path": rep.bad_path,
             "missing": rep.missing}[verdict].append(i)
        return rep

    def audit_ids(self, ids, digest, providers) -> list:
        """Asked about one object or a set: only the trees whose ranges cover them, each id read."""
        reader = Reader(list(self.keys.values()))
        reports = {}
        for obj_id in ids:
            row = reader.row_for(obj_id, digest, providers)
            if row.tree_id not in reports:
                reports[row.tree_id] = self.audit_tree(row, providers)
            rep = reports[row.tree_id]
            try:
                reader.read(obj_id, digest, providers)
            except ReadRefused as e:
                {"ciphertext": rep.bad_ciphertext, "missing": rep.missing}.get(e.kind, rep.bad_path).append(e.position)
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
        """The signed row of the tree that covers obj_id, with a path from the same node to the newest D
        the reader can trust: its own, or a newer one signed by the owner that a node holds."""
        own = crypto.owner_of(obj_id)
        pk = self.keys[own]
        if not verify_digest(pk, digest):
            raise ValueError("D is not signed by this owner")
        v = int.from_bytes(obj_id, "big")
        answers = [p.dht.route(obj_id) for p in providers]          # every node asked, not just one
        newer = [d for *_, d in answers if d is not None and d.version > digest.version and verify_digest(pk, d)]
        d = max(newer, key=lambda x: x.version) if newer else digest
        stale = False
        for row, i, _t, path, _d in answers:
            if row is None or row.own != own or not row.covers(v) or not verify_row(pk, row):
                continue
            if verify_row_in_digest(d, row, i, path):
                return row
            stale = True
        raise KeyError("no node holds a row that reaches the current D" + (" (stale rows only)" if stale else ""))

    def read(self, obj_id, digest, providers) -> bytes:
        """Returns the ciphertext once it reaches the signed root; decryption keys are outside this work."""
        row = self.row_for(obj_id, digest, providers)
        prov = {p.name: p for p in providers}[row.provider]
        tid, version, i, stored, ct, path = prov.read(obj_id)
        if (tid, version) != (row.tree_id, row.version):
            # the provider names another row: fetch the newest again before judging, so a read that
            # crosses a write blames no one, and a provider naming a row no node holds is refused
            fresh = self.row_for(obj_id, digest, providers)
            if (fresh.tree_id, fresh.version) == (tid, version):
                row = fresh
            else:
                raise ReadRefused(row.provider, row.tree_id, i, "wrong tree or version")
        verdict = check_answer(row, i, obj_id, ct, stored, path)
        if verdict != "ok":
            raise ReadRefused(row.provider, row.tree_id, i, verdict)
        return ct
