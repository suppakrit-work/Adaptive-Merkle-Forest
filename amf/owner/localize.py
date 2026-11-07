"""Fault Localization -- Phase 6 and Algorithm 5.

A fault surfaces in three ways: a read or a challenge whose ciphertext does not
hash to the leaf value the provider returns names its own position; a path
whose stored values do not reach the signed root says only that the tree's
stored hashes are wrong; and the owner can ask a provider for a recount, the
root r' it recomputes from the objects it holds right now, or the recomputed
leaf hashes of the whole tree.  Both are the provider's own report: they find
damage a semi-honest provider does not know about, but a malicious one can
answer them from its stored hashes.

Algorithm 5: if the stored root is not r, stop at the tree; if r' = r, clean;
otherwise walk down, round by round, asking for the two children of each node
as a stored value st and a recomputed value rec.  Stored pairs must hash to the
parent already verified, else the stored hashes are damaged; wherever st != rec
the damage is below.  The ranges of the children follow from n alone, so the
owner computes them itself.  The position left is named from the owner's
catalog; the owner reads that object: if H(id || CT) differs from the
catalog's leaf, or the object is gone, it is damaged, otherwise the provider
reported a wrong rec.  A provider that follows the protocol can answer in one
exchange: the positions where stored and recomputed values differ, each with
its stored path, which the owner checks against the signed root.

recount_forest asks every provider at once for (stored root, r') of all the
owner's trees it holds; localize_all then walks the trees that failed, side by
side when the owner runs in parallel.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from amf.common import crypto
from amf.common.merkle import left_size, node_hash, verify_path

READ_HEADER = 16 + 8 + 8 + 32                     # tid, version, position, stored leaf of a read answer


@dataclass
class LocalizationReport:
    tree_id: bytes
    provider: str
    verdict: str          # clean | objects | wrong-rec | stored-hashes-damaged | provider-lied | no-answer
    positions: list = field(default_factory=list)
    names: list = field(default_factory=list)              # the damaged (or lost) objects, by name
    wrong_rec: list = field(default_factory=list)          # positions where P reported a wrong rec
    rounds: int = 0
    owner_hashes: int = 0
    bytes_received: int = 0
    objects_downloaded: int = 0
    note: str = ""


class _Lie(Exception):
    """An answer that breaks the protocol itself."""


class _Damaged(Exception):
    """Stored values that do not hash up to the signed root (Algorithm 5, line 8)."""


class Localizer:
    def __init__(self, owner):
        self.owner = owner

    def name_at(self, row, position: int):
        obj_id = self.owner.catalog.id_at(row.lo, position)
        return obj_id, self.owner.catalog.name_of(obj_id)

    def localize(self, tree_id: bytes, one_exchange: bool = False, roots=None) -> LocalizationReport:
        """roots: (stored root, r') if already asked, e.g. by recount_forest."""
        row = self.owner.index[tree_id]
        prov = self.owner.providers[row.provider]
        own = self.owner.own
        rep = LocalizationReport(tree_id, row.provider, "clean")
        if row.count == 0:
            return rep
        try:
            stored_root, r_prime = roots or (prov.stored_root(own, tree_id), prov.recomputed_root(own, tree_id))
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
            if row.count == 1:
                found = [0]                                      # the root is the leaf: nothing to walk
            else:
                found = self._one_exchange(prov, row, rep) if one_exchange else self._rounds(prov, row, rep)
        except _Damaged as e:
            rep.verdict, rep.note = "stored-hashes-damaged", str(e)
            return rep
        except _Lie as e:
            rep.verdict, rep.note = "provider-lied", str(e)
            return rep
        except Exception as e:
            rep.verdict, rep.note = "no-answer", f"{type(e).__name__}: {e}"
            return rep
        rep.positions = sorted(found)
        for i in rep.positions:                                  # Algorithm 5, lines 14-17
            obj_id, name = self.name_at(row, i)
            try:
                _tid, _v, _i, _stored, ct, path = prov.read(obj_id)
            except Exception as e:
                rep.verdict, rep.note = "no-answer", f"read of {name}: {type(e).__name__}: {e}"
                return rep
            rep.objects_downloaded += 1
            rep.bytes_received += READ_HEADER + len(ct or b"") + 32 * len(path)
            rep.owner_hashes += 1
            if ct is None or crypto.leaf_hash(obj_id, ct) != self.owner.catalog.leaf_of(obj_id):
                rep.names.append(name)
            else:
                rep.wrong_rec.append(i)
        rep.verdict = "objects" if rep.names else "wrong-rec"
        return rep

    def _rounds(self, prov, row, rep) -> list:
        """The round-by-round form: the owner chooses each step (Algorithm 5, lines 4-13), and asks for
        all the nodes of a round in one exchange."""
        todo, found = [(1, 0, row.count, row.root)], []
        while todo:
            rep.rounds += 1
            answers = prov.round_halves(self.owner.own, row.tree_id, [(k, lo, hi) for k, lo, hi, _ in todo])
            if len(answers) != len(todo):
                raise _Lie(f"round {rep.rounds}: {len(answers)} answers for {len(todo)} nodes")
            nxt = []
            for (k, lo, hi, parent), children in zip(todo, answers):
                rep.bytes_received += 4 * 32
                found.extend(self._step(k, lo, hi, parent, children, nxt, rep))
            todo = nxt
        if not found:
            raise _Lie("r' differs from r but no child disagrees")
        return found

    def _step(self, k, lo, hi, parent, children, nxt, rep) -> list:
        mid = lo + left_size(hi - lo)                           # the shape follows from n alone
        (kl, lo_l, hi_l, st_l, rec_l), (kr, lo_r, hi_r, st_r, rec_r) = children
        if (kl, lo_l, hi_l, kr, lo_r, hi_r) != (2 * k, lo, mid, 2 * k + 1, mid, hi):
            raise _Lie(f"node {k}: the halves are not [{lo}, {mid}) and [{mid}, {hi})")
        rep.owner_hashes += 1
        if node_hash(st_l, st_r) != parent:
            raise _Damaged(f"node {k}: stored halves do not hash to the parent already verified")
        found = []
        for kc, loc, hic, st, rec in ((2 * k, lo, mid, st_l, rec_l), (2 * k + 1, mid, hi, st_r, rec_r)):
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
        found = set()
        for i, stored, path in reports:
            rep.bytes_received += 8 + 32 * (1 + len(path))
            if not (isinstance(i, int) and 0 <= i < row.count) or i in found:
                raise _Lie(f"position {i!r} reported out of range or twice")
            rep.owner_hashes += len(path)
            if not verify_path(row.root, row.count, i, stored, path):
                raise _Damaged(f"position {i}: the stored path does not reach the signed root")
            found.add(i)
        if not found:
            raise _Lie("r' differs from r but no position was reported")
        return list(found)

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

    # -- the whole forest at once (III-D, Parallelism) --------------------------------------

    def _each(self, fn, items) -> list:
        if not self.owner.parallel or len(items) < 2:
            return [fn(x) for x in items]
        with ThreadPoolExecutor(max_workers=min(8, len(items))) as pool:
            return list(pool.map(fn, items))

    def recount_forest(self) -> dict:
        """One message to every provider: (stored root, r') of each of the owner's trees it holds.
        Returns tid -> (stored root, r') for the live rows of the owner; a provider that does not
        answer gives (None, None) for its trees."""
        provs = list(self.owner.providers.values())

        def ask(p):
            try:
                return p.recount(self.owner.own)
            except Exception:
                return {}
        answers = dict(zip([p.name for p in provs], self._each(ask, provs)))
        return {r.tree_id: tuple(answers.get(r.provider, {}).get(r.tree_id, (None, None)))
                for r in self.owner.live_rows()}

    def localize_all(self, one_exchange: bool = False) -> list:
        """Recount the forest, then walk every tree whose stored root or r' is not its signed root."""
        roots = self.recount_forest()
        suspects = [tid for tid, (st, rec) in roots.items()
                    if (st, rec) != (self.owner.index[tid].root,) * 2 and self.owner.index[tid].count]
        return self._each(lambda tid: self.localize(tid, one_exchange,
                                                    None if None in roots[tid] else roots[tid]), suspects)
