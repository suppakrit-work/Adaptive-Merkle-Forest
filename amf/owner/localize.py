"""Fault Localization -- Phase 5 and Algorithm 2.

A fault surfaces in three ways: a read or a challenge whose ciphertext does not
hash to the leaf value the provider returns names its own position; a path
whose stored values do not reach the signed root says only that the tree's
stored hashes are wrong; and the owner can ask a provider for a recount, the
root r' it recomputes from the objects it holds right now, or the recomputed
leaf hashes of the whole tree.  Both are the provider's own report: they find
damage a semi-honest provider does not know about, but a malicious one can
answer them from its stored hashes.

Algorithm 2: if the stored root is not r, stop at the tree; if r' = r, clean;
otherwise walk down, round by round, asking for the two children of each node
as a stored value st and a recomputed value rec.  Stored pairs must hash to the
parent already verified; wherever st != rec the damage is below.  The position
left is named from the owner's catalog; the owner reads that object: if
H(id || CT) differs from the catalog's leaf the object is damaged, otherwise the
provider reported a wrong rec.  A provider that follows the protocol can answer
in one exchange: the positions where stored and recomputed values differ, each
with its stored path, which the owner checks against the signed root.
"""
from dataclasses import dataclass, field

from amf.common import crypto
from amf.common.merkle import node_hash, verify_path


@dataclass
class LocalizationReport:
    tree_id: bytes
    provider: str
    verdict: str          # clean | objects | wrong-rec | stored-hashes-damaged | provider-lied | no-answer
    positions: list = field(default_factory=list)
    names: list = field(default_factory=list)              # the damaged objects, by name
    wrong_rec: list = field(default_factory=list)          # positions where P reported a wrong rec
    rounds: int = 0
    owner_hashes: int = 0
    bytes_received: int = 0
    objects_downloaded: int = 0
    note: str = ""


class _Lie(Exception):
    pass


class Localizer:
    def __init__(self, owner):
        self.owner = owner

    def name_at(self, row, position: int):
        obj_id = self.owner.catalog.id_at(row.lo, position)
        return obj_id, self.owner.catalog.name_of(obj_id)

    def localize(self, tree_id: bytes, one_exchange: bool = False) -> LocalizationReport:
        row = self.owner.index[tree_id]
        prov = self.owner.providers[row.provider]
        own = self.owner.own
        rep = LocalizationReport(tree_id, row.provider, "clean")
        if row.count == 0:
            return rep
        try:
            stored_root = prov.stored_root(own, tree_id)
            r_prime = prov.recomputed_root(own, tree_id)
        except Exception as e:                                   # a refusal is charged at the tree
            rep.verdict, rep.note = "no-answer", f"{type(e).__name__}: {e}"
            return rep
        rep.bytes_received += 64
        if stored_root != row.root:
            rep.verdict, rep.note = "stored-hashes-damaged", "stop at the tree"
            return rep
        if r_prime == row.root:
            return rep
        try:
            found = self._one_exchange(prov, row, rep) if one_exchange else self._rounds(prov, row, rep)
        except _Lie as e:
            rep.verdict, rep.note = "provider-lied", str(e)
            return rep
        except Exception as e:
            rep.verdict, rep.note = "no-answer", f"{type(e).__name__}: {e}"
            return rep
        rep.positions = sorted(found)
        for i in rep.positions:                                  # Algorithm 2, lines 14-17
            obj_id, name = self.name_at(row, i)
            _tid, _v, _i, _stored, ct, _path = prov.read(obj_id)
            rep.objects_downloaded += 1
            rep.owner_hashes += 1
            if crypto.leaf_hash(obj_id, ct) != self.owner.catalog.leaf_of(obj_id):
                rep.names.append(name)
            else:
                rep.wrong_rec.append(i)
        rep.verdict = "objects" if rep.names else "wrong-rec"
        return rep

    def _rounds(self, prov, row, rep) -> list:
        """The round-by-round form: the owner chooses each step (Algorithm 2, lines 4-13)."""
        todo, found = [(1, 0, row.count, row.root)], []
        while todo:
            rep.rounds += 1
            nxt = []
            for k, lo, hi, parent in todo:
                children = prov.halves(self.owner.own, row.tree_id, k, lo, hi)
                rep.bytes_received += 4 * 32
                found.extend(self._step(k, lo, hi, parent, children, nxt, rep))
            todo = nxt
        return found

    def _step(self, k, lo, hi, parent, children, nxt, rep) -> list:
        (kl, lo_l, hi_l, st_l, _rl), (kr, lo_r, hi_r, st_r, _rr) = children
        rep.owner_hashes += 1
        if (kl, kr, lo_l, hi_r) != (2 * k, 2 * k + 1, lo, hi) or hi_l != lo_r:
            raise _Lie(f"node {k}: halves do not tile [{lo}, {hi})")
        if node_hash(st_l, st_r) != parent:
            raise _Lie(f"node {k}: stored halves do not hash to the parent already verified")
        found = []
        for kc, loc, hic, st, rec in children:
            if st != rec:
                if hic - loc == 1:
                    found.append(loc)
                else:
                    nxt.append((kc, loc, hic, st))
        return found

    def _one_exchange(self, prov, row, rep) -> list:
        """The provider reports the positions where st and rec differ, each with its stored path;
        the owner checks each path against the signed root, ceil(log2 n) hashes per position."""
        reports = prov.report_mismatches(self.owner.own, row.tree_id)
        rep.rounds = 1
        found = []
        for i, stored, path in reports:
            rep.bytes_received += 32 * (1 + len(path))
            rep.owner_hashes += len(path)
            if not verify_path(row.root, row.count, i, stored, path):
                raise _Lie(f"position {i}: the stored path does not reach the signed root")
            found.append(i)
        if not found:
            raise _Lie("r' differs from r but no position was reported")
        return found

    def recount(self, tree_id: bytes):
        """All n recomputed leaf hashes against the owner's catalog, position by position.
        Returns the positions that disagree.  Trustworthy only against a semi-honest provider."""
        row = self.owner.index[tree_id]
        prov = self.owner.providers[row.provider]
        leaves = prov.recomputed_leaves(self.owner.own, tree_id)
        own = self.owner.catalog.leaves_in(row.lo, row.hi)
        if len(leaves) != len(own):
            return list(range(row.count))
        return [i for i, (a, b) in enumerate(zip(leaves, own)) if a != b]