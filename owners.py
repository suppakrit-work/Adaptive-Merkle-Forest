"""AMF prototype -- many owners sharing three providers (run: python owners.py).

Owners are added in steps (by default until there are 1, 10, 100, 1,000 and
10,000), each with three types and --objects objects, every type in a tree of
its own at one of the providers, so each DHT node holds three rows per owner.
After each step, on owners picked at random, it measures one owner's work:

  upload     owner ms, provider ms and owner hashes for one new object
  read       the owner's read, and a data user's read with D
  audit      one owner: its rows recomputed to D, s challenges per tree
  lookup     a DHT node routing an identifier to its tree, in microseconds,
             and the same by scanning every row (a node without the index)

None of these should grow with the number of other owners.  Owner and provider
ms are split as in bench.py.  The owners are built in bulk as in bench.py (no
encrypted catalog copy at the providers).  Results go to results/owners.csv,
one row per step.
"""
import argparse
import csv
import random
import time

from amf.auditor import Auditor, Reader
from amf.common import crypto
from amf.common.index import IndexRow, id_to_int, new_tree_id, tag_range
from amf.common.merkle import root_of
from amf.owner import Owner
from amf.owner.catalog import TYPES_KEY, Catalog
from amf.provider import Pending, Provider, TreeState
from bench import Clock, med

TYPES = (("image", 1, "A"), ("video", 2, "B"), ("document", 3, "C"))


def add_owner(provs, n_objects, obj_bytes, s, rnd):
    """One owner with its objects and three trees, put in place without one upload per object."""
    o = Owner(s=s, rng=random.Random(rnd.random()))
    for p in provs.values():
        o.register_provider(p)
    o.type_table = {t: tag for t, tag, _ in TYPES}
    entries, by_type = [], {t: [] for t, _, _ in TYPES}
    for k in range(n_objects):
        t = TYPES[k % 3][0]
        obj_id = o.identifier(f"{t}-{k}", t)
        ct = crypto.encrypt(o.mk, obj_id, rnd.randbytes(obj_bytes))
        entries.append((f"{t}-{k}", obj_id, crypto.leaf_hash(obj_id, ct)))
        by_type[t].append((obj_id, ct))
    o.catalog = Catalog.bulk_load(entries)
    rows = []
    for t, tag, prov in TYPES:
        items = sorted(by_type[t])
        lo, hi = tag_range(o.own, tag)
        ids = [i for i, _ in items]
        tid = new_tree_id()
        state = TreeState.build(o.own, tid, lo, hi, ids, [crypto.leaf_hash(i, ct) for i, ct in items])
        root = root_of(o.catalog.leaves_in(lo, hi))             # the owner's own root, from its catalog
        assert root == state.root
        provs[prov].shadow[(o.own, tid)] = Pending(state, "new", staged=dict(items))
        rows.append(IndexRow(o.own, tid, lo, hi, prov, root, len(ids), 1))
    o._publish(rows, {TYPES_KEY: None})
    return o


def scan_lookup(node, obj_id):
    """Routing without the index: look at every row the node holds."""
    v = id_to_int(obj_id)
    return next((r for r in node.rows.values() if not r.retired and r.covers(v)), None)


def a_name(o, n_objects, rnd):
    k = rnd.randrange(n_objects)
    t = TYPES[k % 3][0]
    return f"{t}-{k}", t


def measure(owners, plist, clock, node, lookup, args, rnd):
    def run(o, fn):
        """(client ms, provider ms, owner hashes) of one call."""
        h0, p0, t0 = o.hashes, clock.provider, time.perf_counter()
        fn()
        total, prov = time.perf_counter() - t0, clock.provider - p0
        return 1000 * (total - prov), 1000 * prov, o.hashes - h0

    up, rd, ur, au = [], [], [], []
    for k in range(args.reps):
        o = rnd.choice(owners)
        t = TYPES[k % 3][0]
        data = rnd.randbytes(args.object_bytes)
        up.append(run(o, lambda: o.upload(f"new-{t}-{len(o.catalog)}", data, t)))
        name, t = a_name(o, args.objects, rnd)
        rd.append(run(o, lambda: o.read(name)))
        obj_id, d = o.reader_view(name, t)
        reader, aud, reports = Reader([o.pk]), Auditor([o.pk], s=args.s, rng=random.Random(k)), []
        ur.append(run(o, lambda: reader.read(obj_id, d, plist)))
        au.append(run(o, lambda: reports.append(aud.audit_owner(o.digest, plist))))
        assert reports[0].ok

    def col(xs, j):
        return round(med([x[j] for x in xs]), 4)
    row = {"upload_owner_ms": col(up, 0), "upload_provider_ms": col(up, 1), "upload_hashes": col(up, 2),
           "read_owner_ms": col(rd, 0), "read_provider_ms": col(rd, 1),
           "user_read_ms": col(ur, 0), "user_read_provider_ms": col(ur, 1),
           "audit_ms": col(au, 0), "audit_provider_ms": col(au, 1)}

    ids = []
    for _ in range(args.lookups):
        o = rnd.choice(owners)
        name, _t = a_name(o, args.objects, rnd)
        ids.append(o.catalog.get(name)[0])
    t0 = time.perf_counter()
    found = [lookup(i) for i in ids]
    row["lookup_us"] = round(1e6 * (time.perf_counter() - t0) / len(ids), 3)
    few = ids[:max(1, len(ids) // 10)]
    t0 = time.perf_counter()
    scanned = [scan_lookup(node, i) for i in few]
    row["scan_lookup_us"] = round(1e6 * (time.perf_counter() - t0) / len(few), 3)
    assert None not in found and found[:len(few)] == scanned
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--owners", type=int, nargs="+", default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--objects", type=int, default=30, help="objects per owner, spread over three types")
    ap.add_argument("--object-bytes", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lookups", type=int, default=1000)
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/owners.csv")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    provs = {p: Provider(p) for p in "ABC"}
    plist = list(provs.values())
    lookup = provs["A"].dht.lookup                  # taken before the clock wraps it
    clock = Clock()
    for p in plist:
        clock.watch(p)

    owners, results = [], []
    print(f"{args.objects} objects of {args.object_bytes} B per owner, 3 trees each; medians over {args.reps} runs")
    print(f"{'owners':>7} {'rows/node':>9} {'build s':>8} | {'upload':>14} {'hashes':>6} | {'read':>14} | "
          f"{'user read':>14} | {'audit':>14} | {'lookup us':>9} {'scan us':>9}")
    print(f"{'':>7} {'':>9} {'':>8} | {'owner / prov':>14} {'':>6} | {'owner / prov':>14} | "
          f"{'user / prov':>14} | {'aud. / prov':>14} |")
    for target in sorted(args.owners):
        t0 = time.perf_counter()
        while len(owners) < target:
            owners.append(add_owner(provs, args.objects, args.object_bytes, args.s, rnd))
        row = {"owners": len(owners), "rows_per_node": len(provs["A"].dht.rows),
               "objects": sum(len(p.objects) for p in plist), "build_s": round(time.perf_counter() - t0, 1)}
        row.update(measure(owners, plist, clock, provs["A"].dht, lookup, args, rnd))
        results.append(row)
        r = row
        print(f"{r['owners']:>7,} {r['rows_per_node']:>9,} {r['build_s']:>8} | "
              f"{r['upload_owner_ms']:>6.3f} / {r['upload_provider_ms']:<5.3f} {int(r['upload_hashes']):>6} | "
              f"{r['read_owner_ms']:>6.3f} / {r['read_provider_ms']:<5.3f} | "
              f"{r['user_read_ms']:>6.3f} / {r['user_read_provider_ms']:<5.3f} | "
              f"{r['audit_ms']:>6.3f} / {r['audit_provider_ms']:<5.3f} | "
              f"{r['lookup_us']:>9.2f} {r['scan_lookup_us']:>9.1f}")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()