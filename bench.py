"""AMF prototype -- cost of each operation, and the baselines of Section V (run: python bench.py).

For forests of N objects (one owner, three providers, three types; each
type's objects cut into the fewest trees of at most 3/4 N_up leaves, so 333
per tree at N = 1,000 and about 714 at 10^4 and 10^5) it measures, for a read,
an upload of a new object, a rewrite of a stored one, a split, a merge, a
migration, an audit and a localization:

  owner ms     wall time minus the time spent inside provider calls
  provider ms  time inside provider calls (all three DHT nodes included)
  hashes       hash evaluations the owner spends on its checks
  bytes        what the owner or reader receives to check the operation
               (an announcement is tid 16 + lo 66 + hi 66 + root 32 + count 8)

and the same for the baselines of the baselines/ package (one module per
paper, each saying what it takes from it and what it leaves out):

  single tree  one Merkle tree over all N objects (baselines/wang2011.py):
               read path, provider path update per rewrite, and a rewrite checked
               the Wang way (old path, new root: 2 log2 N hashes)
  rebuild      a reshaping without a check (baselines/mizrahi2024.py, also
               HMT): the verifier downloads the tree and rebuilds the roots
  fetch-all    localization by fetching every leaf hash (wang2011.fetch_all)
  re-challenge localization by challenging halves (baselines/zhu2012.py,
               CPDP): the re-challenges and the blocks the provider touches,
               run on AMF's challenge call and counted

Timed here: every AMF operation (owner and provider sides), the rebuild
baseline, the Wang check and fetch-all.  Counted, not timed: the single-tree
read path and localization depth, and the re-challenge rounds and blocks
(CPDP's tags are not implemented).  The merge timed here joins two neighbors
as they are, whatever their sizes; Table II's rule would only merge two that
sum to under N_low.

The forest is built in bulk (the owner computes every root from its catalog,
signs the rows, the providers promote the trees); the encrypted catalog copy
at the providers is skipped, since recovery is not measured here.  Medians
over --reps runs.  Results go to results/bench.csv, one row per N. 
With --state it measures only what the owner keeps (catalog, node arrays, rows) and writes
results/state.csv instead.
"""
import argparse
import csv
import hashlib
import random
import statistics
import time

from amf.common import crypto
from amf.auditor import Auditor, Reader
from amf.owner.catalog import TYPES_KEY, Catalog
from amf.common.index import (BOUND_LEN, ROW_DOMAIN, SIG_LEN, TREE_ID_LEN, IndexRow, id_to_int, new_tree_id,
                       tag_range)
from amf.owner.localize import Localizer
from amf.common.merkle import depth_of, root_of
from amf.common.merkle import verify_path
from amf.owner import Owner
from amf.provider import Pending, Provider, TreeState
from baselines import mizrahi2024, wang2011, zhu2012

ANNOUNCE = TREE_ID_LEN + 2 * BOUND_LEN + 32 + 8
PROVIDER_CALLS = ("write", "delete", "decompose", "compose", "export_tree", "import_from", "pending_challenge",
                  "release", "receive_batch", "store_catalog_entries", "discard_pending", "acknowledged", "read",
                  "challenge", "recomputed_root", "recomputed_leaves", "stored_root", "stored_leaves", "recount",
                  "halves", "round_halves", "report_mismatches")
DHT_CALLS = ("lookup", "route", "row_with_path", "get")     # the DHT node runs at the provider too


class Clock:
    """Time spent inside provider calls; a call made from inside another one is not counted twice."""

    def __init__(self):
        self.provider = 0.0
        self._inside = 0

    def watch(self, prov):
        for name in PROVIDER_CALLS:
            setattr(prov, name, self._timed(getattr(prov, name)))
        for name in DHT_CALLS:
            setattr(prov.dht, name, self._timed(getattr(prov.dht, name)))

    def _timed(self, fn):
        def run(*a, **k):
            if self._inside:
                return fn(*a, **k)
            self._inside += 1
            t = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                self.provider += time.perf_counter() - t
                self._inside -= 1
        return run


class World:
    def __init__(self, n, obj_bytes, upper, s, seed, single=False):
        self.rnd = random.Random(seed)
        self.owner = Owner(mk=hashlib.sha256(b"amf-bench-%d" % seed).digest(), s=s,   # MK from the seed:
                           rng=random.Random(seed + 1))                              # the same trees every run
        self.provs = {p: Provider(p) for p in "ABC"}
        self.plist = list(self.provs.values())
        self.clock = Clock()
        for p in self.plist:
            self.owner.register_provider(p)
            self.clock.watch(p)
        types = ("all",) if single else ("image", "video", "document")
        self.owner.type_table = {t: k + 1 for k, t in enumerate(types)}
        self.types = types
        self.obj_bytes = obj_bytes
        self._build(n, upper, single)

    def _build(self, n, upper, single):
        o = self.owner
        entries, by_type = [], {t: [] for t in self.types}
        for k in range(n):
            t = self.types[k % len(self.types)]
            name = f"{t}-{k}"
            obj_id = o.identifier(name, t)
            ct = crypto.encrypt(o.mk, obj_id, self.rnd.randbytes(self.obj_bytes))
            entries.append((name, obj_id, crypto.leaf_hash(obj_id, ct)))
            by_type[t].append((obj_id, ct))
        o.catalog = Catalog.bulk_load(entries)
        target = n if single else max(1, 3 * upper // 4)
        rows = []
        for j, t in enumerate(self.types):
            prov = self.plist[j % len(self.plist)]
            items = sorted(by_type[t])
            lo_t, hi_t = tag_range(o.own, o.type_table[t])
            k = max(1, -(-len(items) // target))
            size = -(-len(items) // k) if items else 0
            for c in range(k):
                chunk = items[c * size:(c + 1) * size]
                lo = lo_t if c == 0 else id_to_int(chunk[0][0])
                hi = hi_t if c == k - 1 else id_to_int(items[(c + 1) * size][0])
                ids = [i for i, _ in chunk]
                leaves = o.catalog.leaves_in(lo, hi)
                tid = new_tree_id()
                state = TreeState.build(o.own, tid, lo, hi, ids, [crypto.leaf_hash(i, ct) for i, ct in chunk])
                root = root_of(leaves)                              # the owner's own root, from its catalog
                assert root == state.root and len(leaves) == len(ids)
                prov.shadow[(o.own, tid)] = Pending(state, "new", staged=dict(chunk))
                rows.append(IndexRow(o.own, tid, lo, hi, prov.name, root, len(ids), 1))
        o._publish(rows, {TYPES_KEY: None})
        assert all(not p.shadow for p in self.plist)

    # -- measuring --------------------------------------------------------------

    def run(self, fn):
        """(result, owner ms, provider ms, owner hashes) of one call."""
        h0, p0, t0 = self.owner.hashes, self.clock.provider, time.perf_counter()
        out = fn()
        total = time.perf_counter() - t0
        prov = self.clock.provider - p0
        return out, 1000 * (total - prov), 1000 * prov, self.owner.hashes - h0

    def rows(self):
        return [r for r in self.owner.live_rows() if r.count >= 2]

    def a_tree(self):
        return self.rnd.choice(self.rows())

    def a_name(self, row=None):
        row = row or self.a_tree()
        return self.owner.catalog.name_of(self.owner.catalog.id_at(row.lo, self.rnd.randrange(row.count)))

    def ct_len(self):
        return crypto.IV_LEN + self.obj_bytes + crypto.GCM_TAG_LEN


def med(xs):
    return statistics.median(xs) if xs else 0.0


# ---- AMF ----------------------------------------------------------------------------

def bench_amf(w, reps, s):
    o, out = w.owner, {}
    rows = o.live_rows()
    out["trees"] = len(rows)
    out["mean_tree"] = round(sum(r.count for r in rows) / len(rows), 1)
    out["max_depth"] = max(depth_of(r.count) for r in rows)

    # read, by the owner and by a data user holding id and D; the proof sizes are what one
    # provider (and, for the user, one DHT node) returns, not the bytes of every message
    t_own, p_own, t_user, p_user, path_b, user_b = [], [], [], [], [], []
    reader = Reader([o.pk])
    for _ in range(reps):
        name = w.a_name()
        obj_id = o.catalog.get(name)[0]
        row = o.row_for(obj_id)
        _, om, pm, _ = w.run(lambda: o.read(name))
        t_own.append(om), p_own.append(pm)
        _, um, pm, _ = w.run(lambda: reader.read(obj_id, o.digest, w.plist))
        t_user.append(um), p_user.append(pm)
        path_b.append(32 * depth_of(row.count) + 32)                       # path + stored leaf
        T = o.digest.count
        user_b.append(path_b[-1] + len(row.payload()) - len(ROW_DOMAIN) + SIG_LEN + 32 * depth_of(T))
    out.update(read_owner_ms=med(t_own), read_provider_ms=med(p_own),
               read_user_ms=med(t_user), read_user_provider_ms=med(p_user),
               read_proof_bytes=med(path_b), read_user_proof_bytes=med(user_b))

    # upload of a new object into a tree of at most 3/4 N_up
    om_, pm_, hs, dh = [], [], [], []
    for k in range(reps):
        t = w.rnd.choice(w.types)
        d0 = o.digest_hashes
        _, om, pm, h = w.run(lambda: o.upload(f"{t}-new-{k}", w.rnd.randbytes(w.obj_bytes), t))
        om_.append(om), pm_.append(pm), hs.append(h), dh.append(o.digest_hashes - d0)
    out.update(upload_owner_ms=med(om_), upload_provider_ms=med(pm_), upload_hashes=med(hs),
               upload_d_hashes=med(dh), upload_bytes=ANNOUNCE)

    # rewrite of an object already stored: the path above its leaf, at both ends
    om_, pm_, hs = [], [], []
    for _ in range(reps):
        row = w.a_tree()
        name = w.a_name(row)
        o._tree_of(row)                                       # the owner's nodes of this tree, built once
        _, om, pm, h = w.run(lambda: o.upload(name, w.rnd.randbytes(w.obj_bytes), name.split("-")[0]))
        om_.append(om), pm_.append(pm), hs.append(h)
    out.update(rewrite_owner_ms=med(om_), rewrite_provider_ms=med(pm_), rewrite_hashes=med(hs),
               rewrite_bytes=ANNOUNCE)

    # split at the median and the merge back (two trees of nearly equal size): one hash each
    sm, sp, sh, sd, mm, mp, mh, rb_ms, rb_bytes = [], [], [], [], [], [], [], [], []
    for _ in range(reps):
        row = w.a_tree()
        d0 = o.digest_hashes
        (a, b, _), om, pm, h = w.run(lambda: o.split(row.tree_id))
        sm.append(om), sp.append(pm), sh.append(h), sd.append(o.digest_hashes - d0)
        ab, om, pm, h = w.run(lambda: o.merge(a.tree_id, b.tree_id))
        mm.append(om), mp.append(pm), mh.append(h)
        t = w.provs[ab.provider].live[ab.key]
        r_a, r_b, ms, nbytes = mizrahi2024.rebuild_at_verifier(t.ids, w.provs[ab.provider].objects, a.count)
        assert (r_a, r_b) == (a.root, b.root)
        rb_ms.append(ms), rb_bytes.append(nbytes)
    out.update(split_median_owner_ms=med(sm), split_median_provider_ms=med(sp), split_median_hashes=med(sh),
               split_d_hashes=med(sd), split_bytes=2 * ANNOUNCE,
               merge_equal_owner_ms=med(mm), merge_equal_provider_ms=med(mp), merge_equal_hashes=med(mh),
               merge_bytes=ANNOUNCE, rebuild_verifier_ms=med(rb_ms), rebuild_verifier_bytes=med(rb_bytes))

    # split off the median (a quarter of the way in) and the unequal merge back: recomputed roots
    sm, sp, sh, mm, mp, mh = [], [], [], [], [], []
    for _ in range(reps):
        row = w.a_tree()
        p = id_to_int(o.catalog.id_at(row.lo, max(1, row.count // 4)))
        (a, b, _), om, pm, h = w.run(lambda: o.split(row.tree_id, p))
        sm.append(om), sp.append(pm), sh.append(h)
        _, om, pm, h = w.run(lambda: o.merge(a.tree_id, b.tree_id))
        mm.append(om), mp.append(pm), mh.append(h)
    out.update(split_other_owner_ms=med(sm), split_other_provider_ms=med(sp), split_other_hashes=med(sh),
               merge_unequal_owner_ms=med(mm), merge_unequal_provider_ms=med(mp), merge_unequal_hashes=med(mh))

    # migration: root unchanged, s challenges at the destination; then the release order
    mo, mp, mh, mb = [], [], [], []
    for _ in range(reps):
        row = w.a_tree()
        dest = w.rnd.choice([p for p in w.provs if p != row.provider])
        _, om, pm, h = w.run(lambda: o.migrate(row.tree_id, dest))
        o.release(row.tree_id, row.provider)
        mo.append(om), mp.append(pm), mh.append(h)
        k = min(s, row.count)
        mb.append(ANNOUNCE + k * (crypto.ID_LEN + w.ct_len() + 32 + 32 * depth_of(row.count)))
    out.update(migrate_owner_ms=med(mo), migrate_provider_ms=med(mp), migrate_hashes=med(mh),
               migrate_bytes=med(mb))

    # audit of one tree by an auditor holding public keys only
    aud = Auditor([o.pk], s=s, rng=random.Random(7))
    am, ap, ab_ = [], [], []
    for _ in range(reps):
        row = w.a_tree()
        _, ms, pm, _ = w.run(lambda: aud.audit_tree(row, w.plist))
        am.append(ms), ap.append(pm)
        ab_.append(min(s, row.count) * (crypto.ID_LEN + w.ct_len() + 32 + 32 * depth_of(row.count)))
    out.update(audit_ms=med(am), audit_provider_ms=med(ap), audit_bytes=med(ab_))

    out.update(bench_localize(w, reps))
    return out


def bench_localize(w, reps):
    o, loc = w.owner, Localizer(w.owner)
    r_ms, r_pm, rounds, hashes, by, e_ms, e_h, e_b, fa_ms, fa_pm, fa_b, rc_rounds, rc_blocks = ([] for _ in range(13))
    for _ in range(reps):
        row = w.a_tree()
        prov = w.provs[row.provider]
        pos = w.rnd.randrange(row.count)
        victim = o.catalog.id_at(row.lo, pos)
        prov.corrupt(victim, at=20)
        rep, om, pm, _ = w.run(lambda: loc.localize(row.tree_id))
        assert rep.positions == [pos] and rep.verdict == "objects"
        r_ms.append(om), r_pm.append(pm), rounds.append(rep.rounds), hashes.append(rep.owner_hashes)
        by.append(rep.bytes_received)
        one, om, _, _ = w.run(lambda: loc.localize(row.tree_id, one_exchange=True))
        e_ms.append(om), e_h.append(one.owner_hashes), e_b.append(one.bytes_received)
        (found, nbytes), om, pm, _ = w.run(                                   # the owner's leaves come from its catalog
            lambda: wang2011.fetch_all(prov, o.own, row.tree_id, o.catalog.leaves_in(row.lo, row.hi)))
        assert found == [pos]
        fa_ms.append(om), fa_pm.append(pm), fa_b.append(nbytes)

        def check_range(lo, hi):                                             # one CPDP-style challenge
            for i, obj_id, ct, _, path in prov.challenge(o.own, row.tree_id, range(lo, hi)):
                if ct is None or not verify_path(row.root, row.count, i, crypto.leaf_hash(obj_id, ct), path):
                    return False
            return True
        where, rounds_, blocks = zhu2012.localize(check_range, row.count)
        d = depth_of(row.count)                                              # a leaf's path is d or d - 1 long
        assert where == pos and d - 1 <= rounds_ <= d and blocks <= zhu2012.expected(row.count)[1]
        rc_rounds.append(rounds_), rc_blocks.append(blocks)
        prov.corrupt(victim, at=20)                                          # flip the byte back
    return dict(localize_owner_ms=med(r_ms), localize_provider_ms=med(r_pm), localize_rounds=med(rounds),
                localize_hashes=med(hashes), localize_bytes=med(by), one_exchange_owner_ms=med(e_ms),
                one_exchange_hashes=med(e_h), one_exchange_bytes=med(e_b), fetch_all_owner_ms=med(fa_ms),
                fetch_all_provider_ms=med(fa_pm), fetch_all_bytes=med(fa_b), rechallenge_rounds=med(rc_rounds), rechallenge_blocks=med(rc_blocks))


# ---- baseline: one Merkle tree over all objects ---------------------------------------------

def bench_single(w, reps):
    o, out = w.owner, {}
    (row,) = o.live_rows()
    t = w.provs[row.provider].live[row.key]
    out["single_depth"] = wang2011.depth(row.count)
    out["single_read_proof_bytes"] = wang2011.proof_bytes(row.count)
    # a rewrite checked the Wang way: the old leaf and path give the old root, the new leaf the new root
    wm, wh, picked = [], [], []
    for _ in range(reps):
        i = w.rnd.randrange(row.count)
        picked.append(i)
        old, path = t.leaves[i], t.tree.path(i)
        new = crypto.leaf_hash(t.ids[i], w.rnd.randbytes(w.ct_len()))
        check = wang2011.PathUpdate()
        t0 = time.perf_counter()
        check.check(row.root, row.count, i, old, new, path)
        wm.append(1000 * (time.perf_counter() - t0))
        wh.append(check.hashes)
    out.update(single_rewrite_check_ms=med(wm), single_rewrite_check_hashes=med(wh))
    # the same rewrites under AMF's check on the one tree, at the same positions: the path at both
    # ends (o.hashes; D is not counted here)
    pm_, om_, hs = [], [], []
    o._tree_of(row)                                          # the owner's nodes of the tree, built once
    for i in picked:
        name = w.owner.catalog.name_of(t.ids[i])
        _, om, pm, h = w.run(lambda: o.upload(name, w.rnd.randbytes(w.obj_bytes), "all"))
        pm_.append(pm), om_.append(om), hs.append(h)
    out.update(single_write_provider_ms=med(pm_), single_write_owner_ms=med(om_), single_write_hashes=med(hs))
    out["single_localize_rounds"] = depth_of(row.count)
    out["single_fetch_all_bytes"] = wang2011.fetch_all_bytes(row.count)
    return out


def owner_state(n, obj_bytes, upper, s, seed):
    """What the owner keeps for n objects: the catalog (identifier, leaf hash and file name per object),
    the node arrays of its trees (all of them built, so an upper bound) and its copy of the rows."""
    w = World(n, obj_bytes, upper, s, seed)
    o = w.owner
    live = o.live_rows()
    for r in live:
        o._tree_of(r)
    cat = len(o.catalog) * (crypto.ID_LEN + 32)
    names = sum(len(name.encode()) for name, _, _ in o.catalog.entries())
    nodes = sum(len(t._nodes) for t in o._trees.values())
    rows = sum(len(r.payload()) + len(r.signature) for r in live)
    total = cat + names + nodes + rows
    return {"n": n, "trees": len(live), "catalog_bytes": cat, "name_bytes": names, "node_bytes": nodes,
            "row_bytes": rows, "total_bytes": total, "bytes_per_object": round(total / n, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    ap.add_argument("--object-bytes", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--upper", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/bench.csv")
    ap.add_argument("--state", action="store_true", help="only measure the owner's state, into results/state.csv")
    args = ap.parse_args()

    if args.state:
        rows = [owner_state(n, args.object_bytes, args.upper, args.s, args.seed) for n in args.sizes]
        for r in rows:
            print(r)
        with open("results/state.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("\nwrote results/state.csv")
        return

    results = []
    for n in args.sizes:
        t0 = time.perf_counter()
        w = World(n, args.object_bytes, args.upper, args.s, args.seed)
        row = {"n": n, "object_bytes": args.object_bytes, "s": args.s, "n_up": args.upper,
               "build_s": round(time.perf_counter() - t0, 1)}
        row.update(bench_amf(w, args.reps, args.s))
        del w
        w = World(n, args.object_bytes, args.upper, args.s, args.seed, single=True)
        row.update(bench_single(w, args.reps))
        del w
        results.append({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        print_row(results[-1])

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(results[0]))
        wr.writeheader()
        wr.writerows(results)
    print(f"\nwrote {args.out}")


def print_row(r):
    print(f"\nN = {r['n']:,}: {r['trees']} trees, mean {r['mean_tree']} leaves, depth <= {r['max_depth']}; "
          f"objects {r['object_bytes']} B, s = {r['s']} (built in {r['build_s']} s)")
    print(f"  {'operation':28} {'owner ms':>9} {'prov ms':>9} {'hashes':>7} {'bytes':>9}"
          f"   (keeping D: {int(r['upload_d_hashes'])} hashes for a row changed in place, "
          f"{int(r['split_d_hashes'])} when rows are added)")
    lines = [
        ("read (owner)", r["read_owner_ms"], r["read_provider_ms"], None, r["read_proof_bytes"]),
        ("read (data user, with D)", r["read_user_ms"], r["read_user_provider_ms"], None,
         r["read_user_proof_bytes"]),
        ("upload (new object)", r["upload_owner_ms"], r["upload_provider_ms"], r["upload_hashes"], r["upload_bytes"]),
        ("rewrite (path update)", r["rewrite_owner_ms"], r["rewrite_provider_ms"], r["rewrite_hashes"],
         r["rewrite_bytes"]),
        ("split at the median", r["split_median_owner_ms"], r["split_median_provider_ms"],
         r["split_median_hashes"], r["split_bytes"]),
        ("split elsewhere", r["split_other_owner_ms"], r["split_other_provider_ms"], r["split_other_hashes"],
         r["split_bytes"]),
        ("merge, near-equal", r["merge_equal_owner_ms"], r["merge_equal_provider_ms"], r["merge_equal_hashes"],
         r["merge_bytes"]),
        ("merge, unequal", r["merge_unequal_owner_ms"], r["merge_unequal_provider_ms"], r["merge_unequal_hashes"],
         r["merge_bytes"]),
        ("migration (s challenges)", r["migrate_owner_ms"], r["migrate_provider_ms"], r["migrate_hashes"],
         r["migrate_bytes"]),
        ("audit of one tree", r["audit_ms"], r["audit_provider_ms"], None, r["audit_bytes"]),
        ("localize, rounds", r["localize_owner_ms"], r["localize_provider_ms"], r["localize_hashes"],
         r["localize_bytes"]),
        ("localize, one exchange", r["one_exchange_owner_ms"], None, r["one_exchange_hashes"],
         r["one_exchange_bytes"]),
    ]
    for name, om, pm, h, b in lines:
        print(f"  {name:28} {om:>9.3f} {'' if pm is None else f'{pm:9.3f}':>9} "
              f"{'' if h is None else int(h):>7} {int(b):>9,}")
    print("  baselines")
    print(f"  {'rebuild at the verifier':28} {r['rebuild_verifier_ms']:>9.3f} {'':>9} {'':>7} "
          f"{int(r['rebuild_verifier_bytes']):>9,}")
    print(f"  {'localize, fetch all leaves':28} {r['fetch_all_owner_ms']:>9.3f} {r['fetch_all_provider_ms']:>9.3f} {'':>7} "
          f"{int(r['fetch_all_bytes']):>9,}")
    print(f"  {'localize, re-challenge':28} {'':>9} {'':>9} {'':>7} {'':>9}   "
          f"{int(r['rechallenge_rounds'])} challenges, {int(r['rechallenge_blocks']):,} blocks touched")
    print(f"  {'single tree: read path':28} {'':>9} {'':>9} {'':>7} {int(r['single_read_proof_bytes']):>9,}"
          f"   depth {r['single_depth']}")
    print(f"  {'single tree: rewrite (Wang)':28} {r['single_rewrite_check_ms']:>9.3f} {'':>9} "
          f"{int(r['single_rewrite_check_hashes']):>7} {'':>9}")
    print(f"  {'single tree: rewrite (ours)':28} {r['single_write_owner_ms']:>9.3f} "
          f"{r['single_write_provider_ms']:>9.3f} {int(r['single_write_hashes']):>7} {'':>9}   the path at both ends")
    print(f"  {'single tree: localize':28} {'':>9} {'':>9} {'':>7} {int(r['single_fetch_all_bytes']):>9,}"
          f"   {r['single_localize_rounds']} rounds, or every leaf hash")


if __name__ == "__main__":
    main()
