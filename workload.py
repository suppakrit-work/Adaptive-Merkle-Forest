"""AMF prototype -- does adapting pay?  One workload, two forests and one tree (run: python workload.py).

The same forest of N objects (trees of at most 3/4 N_up, as in bench.py) is
driven by the same sequence of accesses under two configurations:

  static   a forest with the same size bounds that never restructures
  AMF      the Policy Manager of Phase 4 with the values of Table II, one
           policy round after every --per-round accesses

and, for reference, one tree over all N objects, computed rather than run
(every access lands in that one tree, so its costs follow from N alone):
with the owner's check of Algorithm 3 (n - 1 hashes, the provider rebuilds
all n leaves), and with a path update as in Wang et al. (2 log2 N hashes at
the verifier, log2 N at the provider).

An access is a read or, with probability --writes, a rewrite of an existing
object.  Objects are ranked by a Zipf law of exponent a (a = 0 is uniform);
since identifiers are HMACs, popular objects are scattered over the trees.
Reported per access: the owner's hashes (checks and keeping D), the leaves
the provider rebuilds, the read path, owner and provider time; and the
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
from amf.common.merkle import depth_of, sides_of
from amf.owner import Owner
from amf.owner.policy import PolicyManager
from amf.provider import Provider

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
    pm = PolicyManager(o) if config == "AMF" else None    # Table II: 1000, 250, 4, 50, 3
    names = {k: (f"{w.types[k % 3]}-{k}", w.types[k % 3]) for k in range(args.n)}
    tot = dict(owner_hashes=0, provider_leaves=0, read_path=0, reads=0, writes=0,
               owner_ms=0.0, provider_ms=0.0, splits=0, merges=0)
    o.hashes = o.digest_hashes = 0
    for step, (obj, is_write) in enumerate(seq, start=1):
        name, ftype = names[obj]
        obj_id = o.catalog.get(name)[0]
        row = o.row_for(obj_id)
        p0, t0 = w.clock.provider, time.perf_counter()
        if is_write:
            o.upload(name, w.rnd.randbytes(w.obj_bytes), ftype)
            tot["writes"] += 1
            tot["provider_leaves"] += row.count                    # the provider rebuilds the whole tree
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


def single_tree(n, seq, wang):
    """One tree over all n objects: every access lands in it, so its costs follow from n."""
    reads = sum(1 for _, wr in seq if not wr)
    writes = len(seq) - reads
    d = depth_of(n)
    if wang:                                              # verify the old path, compute the new root
        owner, prov = writes * 2 * d, writes * d
    else:                                                 # Algorithm 3 on one tree of n leaves
        owner, prov = writes * (n - 1), writes * n
    return dict(owner_hashes=owner, provider_leaves=prov, read_path=reads * d, reads=reads, writes=writes,
                owner_ms=None, provider_ms=None, splits=0, merges=0, trees=1, mean_tree=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--accesses", type=int, default=10000)
    ap.add_argument("--per-round", type=int, default=1000, help="accesses between two policy rounds")
    ap.add_argument("--object-bytes", type=int, default=1024)
    ap.add_argument("--upper", type=int, default=1000)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/workload.csv")
    args = ap.parse_args()

    out, summary = [], []
    for wl, (a, writes) in WORKLOADS.items():
        seq = accesses(args.n, a, writes, args.accesses, args.seed)
        res = {"never restructures": run("static", args, seq), "AMF": run("AMF", args, seq),
               "one tree, Algorithm 3": single_tree(args.n, seq, wang=False),
               "one tree, path update": single_tree(args.n, seq, wang=True)}
        show(wl, args, res)
        summary.append((wl, res["never restructures"], res["AMF"]))
        for config, t in res.items():
            out.append({"workload": wl, "zipf": a, "writes": writes, "n": args.n, "accesses": len(seq),
                        "config": config, **per_access(t, len(seq))})

    print(f"\nAMF against the forest that never restructures (N = {args.n:,}, {args.accesses:,} accesses)")
    print(f"  {'workload':24} {'splits':>6} {'owner hashes':>13} {'prov leaves':>12} {'read path':>10}")
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
    print(f"  {'per access':24} {'owner hashes':>12} {'prov leaves':>11} {'read path':>9} {'owner ms':>9} "
          f"{'prov ms':>8} {'splits':>6} {'trees':>5}")
    for config, t in res.items():
        p = per_access(t, args.accesses)
        ms = "" if p["owner_ms"] is None else f"{p['owner_ms']:9.3f} {p['provider_ms']:8.3f}"
        print(f"  {config:24} {p['owner_hashes']:>12,.1f} {p['provider_leaves']:>11,.1f} {p['read_path']:>9.2f} "
              f"{ms:>18} {p['splits']:>6} {p['trees']:>5}")


if __name__ == "__main__":
    main()
