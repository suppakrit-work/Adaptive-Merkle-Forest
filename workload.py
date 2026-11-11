"""AMF prototype -- does adapting pay?  One workload, two forests and one tree (run: python workload.py).

The same forest of N objects (trees of at most 3/4 N_up, as in bench.py) is
driven by the same sequence of accesses under two configurations:

  static   a forest with the same size bounds that never restructures
           (baselines/amf_static.py)
  AMF      the Policy Manager of Phase 4 with the values of Table II, one
           policy round after every --per-round accesses

and, for reference, one tree over all N objects, computed rather than run
(every access lands in that one tree, so its costs follow from N alone):
checked as AMF checks a rewrite (the path above the leaf and one hash for D;
amf_static.one_tree_costs), and with a path update as in Wang et al. (2 log2
N hashes at the verifier, log2 N at the provider; baselines/wang2011.py).

HMT, the frequency-aware tiered tree of Shangguan et al. (baselines/
shangguan2026.py), runs on the same sequence as a fifth configuration: the
same number N of leaves (random 32-byte values under uniform 32-byte keys,
since its tries need uniform keys; the hash counts do not depend on the
values), a period every --hmt-period accesses standing for a block, theta =
0.5 accesses per period, a hot tier of N/10 and N/5 candidates, the hot tier
rebuilt every 5 periods (the paper's evaluation uses theta = 0.05, caches of
8,000/16,000 and a rebuild every 500 blocks; with B = 1 the Ratio-Based
test admits any key accessed once in the first period, the paper's
policy).  Its "owner hashes" are what a verifier spends checking rewrites by
the path update while trusting every layout change (HMT, layout trusted), or
recomputing the tiers whose root changed (HMT, layout rebuilt); read checks
are left out of every row's owner hashes (Owner.read counts none) and show
in the read-path column; its provider column is the hashes of the
maintainer; read path is the access-weighted proof length.

An access is a read or, with probability --writes, a rewrite of an existing
object.  Objects are ranked by a Zipf law of exponent a (a = 0 is uniform);
since identifiers are HMACs, popular objects are scattered over the trees.
Reported per access: the owner's hashes (checks and keeping D), the hashes
at the provider (the path above a rewritten leaf; every leaf of the trees a
split or merge builds), the read path, owner and provider time; and the
restructurings the policy made, whose own cost is included in the totals.
Hash and leaf counts barely move from run to run (tree identifiers are
random, so a path in D can differ by a hash); times depend on the machine.
Results go to results/workload.csv.
"""
import argparse
import csv
import hashlib
import itertools
import random
import time

from bench import Clock, World
from amf.common.merkle import sides_of
from amf.owner import Owner
from amf.owner.policy import PolicyManager
from amf.provider import Provider
from baselines import amf_static, shangguan2026, wang2011

WORKLOADS = {                       # name: (Zipf exponent, share of writes)
    "uniform, 10% writes": (0.0, 0.1),
    "zipf 1.0, 10% writes": (1.0, 0.1),
    "zipf 0.8, 50% writes": (0.8, 0.5),
    "zipf 1.0, 50% writes": (1.0, 0.5),
    "zipf 1.2, 50% writes": (1.2, 0.5),
    "zipf 1.5, 50% writes": (1.5, 0.5),
}


class TraceWorld(World):
    """bench.World with a master key fixed by the seed, so every configuration has the same trees."""

    def __init__(self, n, obj_bytes, upper, s, seed):
        self.rnd = random.Random(seed)
        mk = hashlib.sha256(b"amf-trace-%d" % seed).digest()
        self.owner = Owner(mk=mk, s=s, rng=random.Random(seed + 1))
        self.provs = {p: Provider(p) for p in "ABC"}
        self.plist = list(self.provs.values())
        self.clock = Clock()
        for p in self.plist:
            self.owner.register_provider(p)
            self.clock.watch(p)
        self.types = ("image", "video", "document")
        self.owner.type_table = {t: k + 1 for k, t in enumerate(self.types)}
        self.obj_bytes = obj_bytes
        self._build(n, upper, single=False)


def accesses(n, a, writes, total, seed):
    """The access sequence: (object number, is_write), the same for every configuration."""
    rnd = random.Random(seed)
    ranked = list(range(n))
    rnd.shuffle(ranked)                                   # rank k -> object ranked[k]
    cum = list(itertools.accumulate(1.0 / (k + 1) ** a for k in range(n)))
    picks = rnd.choices(ranked, cum_weights=cum, k=total)
    return [(obj, rnd.random() < writes) for obj in picks]


def run(config, args, seq):
    w = TraceWorld(args.n, args.object_bytes, args.upper, args.s, args.seed)
    o = w.owner
    pm = PolicyManager(o) if config == "AMF" else amf_static.no_policy(o)    # Table II: 1000, 250, 4, 50, 3
    names = {k: (f"{w.types[k % 3]}-{k}", w.types[k % 3]) for k in range(args.n)}
    tot = dict(owner_hashes=0, provider_leaves=0, read_path=0, reads=0, writes=0,
               owner_ms=0.0, provider_ms=0.0, splits=0, merges=0)
    for r in o.live_rows():                # the node arrays the owner keeps, built once before the trace (as
        o._tree_of(r)                      # bench.py does); a split leaves its halves without arrays, so the
                                           # first rewrite into each rebuilds n_half - 1: a price of restructuring
    o.hashes = o.digest_hashes = 0
    for step, (obj, is_write) in enumerate(seq, start=1):
        name, ftype = names[obj]
        obj_id = o.catalog.get(name)[0]
        row = o.row_for(obj_id)
        p0, t0 = w.clock.provider, time.perf_counter()
        if is_write:
            o.upload(name, w.rnd.randbytes(w.obj_bytes), ftype)
            tot["writes"] += 1
            tot["provider_leaves"] += len(sides_of(row.count, o.catalog.position(obj_id, row.lo)))
            #                         a rewrite: the provider rehashes the path above the leaf
        else:
            o.read(name)
            tot["reads"] += 1
            tot["read_path"] += len(sides_of(row.count, o.catalog.position(obj_id, row.lo)))
        spent = time.perf_counter() - t0
        tot["provider_ms"] += 1000 * (w.clock.provider - p0)
        tot["owner_ms"] += 1000 * (spent - (w.clock.provider - p0))
        if pm and step % args.per_round == 0:
            before = {r.tree_id: r.count for r in o.live_rows()}
            p0, t0 = w.clock.provider, time.perf_counter()
            for d in pm.tick():
                if d.kind == "split":
                    tot["splits"] += 1
                    tot["provider_leaves"] += before[d.trees[0]]   # decompose builds both halves
                else:
                    tot["merges"] += 1
                    tot["provider_leaves"] += sum(before[t] for t in d.trees)
            spent = time.perf_counter() - t0
            tot["provider_ms"] += 1000 * (w.clock.provider - p0)
            tot["owner_ms"] += 1000 * (spent - (w.clock.provider - p0))
    tot["owner_hashes"] = o.hashes + o.digest_hashes
    rows = o.live_rows()
    tot["trees"] = len(rows)
    tot["mean_tree"] = round(sum(r.count for r in rows) / len(rows), 1)
    return tot


def run_hmt(args, seq):
    """The same objects and accesses on HMT, with two verifiers watching: one trusts every
    layout change, one recomputes the tiers.  Returns both as configurations of the usual shape."""
    rnd = random.Random(args.seed)
    keys = [hashlib.sha256(b"amf-trace-%d-%d" % (args.seed, k)).digest() for k in range(args.n)]
    leaves = [rnd.randbytes(32) for _ in range(args.n)]
    h = shangguan2026.HMT(zip(keys, leaves), theta=0.5, hot_capacity=max(1, args.n // 10),
                          candidates=max(1, args.n // 5), rebuild_every=5)
    trusting, rebuilding = shangguan2026.Verifier(h.global_root), shangguan2026.Verifier(h.global_root)
    m0, ver_ms, mnt_ms, rb_ms = h.maintainer.n, 0.0, 0.0, 0.0
    plen = reads = writes = read_hashes = 0
    for step, (obj, is_write) in enumerate(seq, start=1):
        k = keys[obj]
        if is_write:
            old, new = leaves[obj], rnd.randbytes(32)
            t0 = time.perf_counter()
            proof, tier, roots = h.read(k, track=False)          # the path the update is checked on; write counts the access
            h.write(k, new)
            mnt_ms += 1000 * (time.perf_counter() - t0)
            t0 = time.perf_counter()
            assert trusting.check_write(old, new, proof, tier, roots)
            ver_ms += 1000 * (time.perf_counter() - t0)
            assert rebuilding.check_write(old, new, proof, tier, roots)
            assert trusting.global_root == h.global_root
            leaves[obj] = new
            writes += 1
        else:
            t0 = time.perf_counter()
            proof, tier, roots = h.read(k)
            mnt_ms += 1000 * (time.perf_counter() - t0)
            t0, h0 = time.perf_counter(), trusting.hashes
            assert trusting.check_read(leaves[obj], proof, tier, roots)
            ver_ms += 1000 * (time.perf_counter() - t0)
            read_hashes += trusting.hashes - h0      # a read check: in no AMF row (owner.read counts none), so left out here too
            assert rebuilding.check_read(leaves[obj], proof, tier, roots)
            plen += len(proof) + (2 if tier != "cold" and roots[3] else 1)
            reads += 1
        if step % args.hmt_period == 0:
            held = h.roots()                                    # what the verifiers hold going into the period
            t0 = time.perf_counter()
            r = h.end_of_period()
            mnt_ms += 1000 * (time.perf_counter() - t0)
            if r["changed"]:
                trusting.trust_layout(h.global_root)
                t0 = time.perf_counter()
                rebuilding.rebuild_layout(h, held)
                rb_ms += 1000 * (time.perf_counter() - t0)
    sizes = h.tier_sizes()
    base = dict(provider_leaves=h.maintainer.n - m0, read_path=plen, reads=reads, writes=writes,
                owner_ms=ver_ms, provider_ms=mnt_ms, splits=h.stats["promoted"], merges=h.stats["rebuilds"],
                trees=2, mean_tree=sizes["base"] + sizes["overflow"])   # HMT: splits = promotions, merges = rebuilds, mean_tree = hot tier
    return {"HMT, layout trusted": dict(base, owner_hashes=trusting.hashes - read_hashes),
            "HMT, layout rebuilt": dict(base, owner_hashes=rebuilding.hashes - read_hashes, owner_ms=ver_ms + rb_ms)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--accesses", type=int, default=10000)
    ap.add_argument("--per-round", type=int, default=1000, help="accesses between two policy rounds")
    ap.add_argument("--object-bytes", type=int, default=1024)
    ap.add_argument("--upper", type=int, default=1000)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hmt-period", type=int, default=200, help="accesses per HMT period (policy runs at its end)")
    ap.add_argument("--out", default="results/workload.csv")
    args = ap.parse_args()

    out, summary = [], []
    for wl, (a, writes) in WORKLOADS.items():
        seq = accesses(args.n, a, writes, args.accesses, args.seed)
        reads = sum(1 for _, wr in seq if not wr)
        res = {"never restructures": run("static", args, seq), "AMF": run("AMF", args, seq),
               "one tree, AMF check": amf_static.one_tree_costs(args.n, reads, len(seq) - reads),
               "one tree, path update": wang2011.access_costs(args.n, reads, len(seq) - reads),
               **run_hmt(args, seq)}
        show(wl, args, res)
        summary.append((wl, res["never restructures"], res["AMF"]))
        for config, t in res.items():
            out.append({"workload": wl, "zipf": a, "writes": writes, "n": args.n, "accesses": len(seq),
                        "config": config, **per_access(t, len(seq))})

    print(f"\nAMF against the forest that never restructures (N = {args.n:,}, {args.accesses:,} accesses)")
    print(f"  {'workload':24} {'splits':>6} {'owner hashes':>13} {'prov hashes':>12} {'read path':>10}")
    for wl, s, a in summary:
        print(f"  {wl:24} {a['splits']:>6} {change(s, a, 'owner_hashes'):>13} "
              f"{change(s, a, 'provider_leaves'):>12} "
              f"{s['read_path'] / s['reads']:>5.2f} -> {a['read_path'] / a['reads']:.2f}")

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"\nwrote {args.out}")


def change(s, a, k):
    return f"{round(100 * (a[k] - s[k]) / s[k], 1) + 0.0:+.1f}%" if s[k] else "-"


def per_access(t, total):
    def each(k, digits=2):
        return None if t[k] is None else round(t[k] / total, digits)
    return {"owner_hashes": each("owner_hashes", 1), "provider_leaves": each("provider_leaves", 1),
            "read_path": round(t["read_path"] / t["reads"], 2) if t["reads"] else 0,
            "owner_ms": each("owner_ms", 4), "provider_ms": each("provider_ms", 4),
            "splits": t["splits"], "merges": t["merges"], "trees": t["trees"], "mean_tree": t["mean_tree"]}


def show(wl, args, res):
    print(f"\n{wl}: N = {args.n:,}, {args.accesses:,} accesses, a policy round every {args.per_round:,}")
    print(f"  {'per access':24} {'owner hashes':>12} {'prov hashes':>11} {'read path':>9} {'owner ms':>9} "
          f"{'prov ms':>8} {'splits':>6} {'trees':>5}   (HMT: splits = promotions, merges = rebuilds, trees = tiers)")
    for config, t in res.items():
        p = per_access(t, args.accesses)
        ms = "" if p["owner_ms"] is None else f"{p['owner_ms']:9.3f} {p['provider_ms']:8.3f}"
        print(f"  {config:24} {p['owner_hashes']:>12,.1f} {p['provider_leaves']:>11,.1f} {p['read_path']:>9.2f} "
              f"{ms:>18} {p['splits']:>6} {p['trees']:>5}")


if __name__ == "__main__":
    main()
