"""AMF over a real network: the owner, a data user and an auditor against running provider services.

Start the providers first (docker compose up, or python -m amf.provider.server ...), then

    python net_demo.py                                   # the three containers on this machine
    python net_demo.py --provider A=http://1.2.3.4:8000 --provider B=... --provider C=...
    python net_demo.py --files 10000 --batch 200 --reads 2000 --zipf 1.0 --rounds 3

The owner runs here; each provider is reached over HTTP.  Every step is timed
and the bytes and round trips it took are counted at the owner's end, and the
wall time is split into the owner's own CPU time (checks, signatures, the
wire format) and the time spent waiting on the providers, so the same script
measures a run on localhost or across three clouds.  Objects go up as batch
writes of --batch objects per tree (one announcement, one check, Algorithm 3
line 12), then single writes and reads are timed on their own.  Reads follow
a Zipf law over the objects (--zipf 0 for uniform); --locality F sends nine
reads in ten into the lowest fraction F of each type's identifier range, a
hot range the policy can see.  The Policy Manager runs --rounds policy rounds
with the values of Table II, and each round is timed with its orders out
together.  With
providers started in --debug mode it also damages one object, names it, and
moves the damaged provider's trees away.  The owner sends each new row to
all DHT nodes at once (--sequential: one after another).  Results go to
results/net.csv.
"""
import argparse
import csv
import random
import statistics
import time

from amf.auditor import Auditor, Reader
from amf.common.index import id_to_int, tag_range
from amf.owner.localize import Localizer
from amf.owner import Owner, ReadFault
from amf.owner.policy import PolicyManager
from amf.provider.client import RemoteError, RemoteProvider

TYPES = (("image", 1, "A"), ("video", 2, "B"), ("document", 3, "C"))


class Meter:
    """Wall time, time on the wire, round trips and bytes of one step, summed over every provider."""

    def __init__(self, provs):
        self.provs = provs
        self.rows = []

    def counters(self):
        return [sum(getattr(p.net, k) for p in self.provs) for k in ("calls", "sent", "received")]

    def run(self, step, fn, repeat=1):
        times, cpus, out = [], [], None
        for k in range(repeat):
            c0 = self.counters()
            t0, p0 = time.perf_counter(), time.process_time()
            out = fn()
            times.append(1000 * (time.perf_counter() - t0))
            cpus.append(1000 * (time.process_time() - p0))
            if k == 0:
                first = [b - a for a, b in zip(c0, self.counters())]
        wall, cpu = statistics.median(times), statistics.median(cpus)
        row = {"step": step, "runs": repeat, "median_ms": round(wall, 2), "owner_cpu_ms": round(min(cpu, wall), 2),
               "wait_ms": round(max(wall - cpu, 0), 2), "calls": first[0], "sent_bytes": first[1],
               "received_bytes": first[2]}
        self.rows.append(row)
        print(f"  {step:38} {row['median_ms']:>9.2f} ms  cpu {row['owner_cpu_ms']:>8.2f}  wait {row['wait_ms']:>8.2f}"
              f"  {row['calls']:>5} calls {row['sent_bytes']:>9,} B out {row['received_bytes']:>9,} B in")
        return out


def wait_for(p, seconds=30):
    """A container that has just started may need a moment before it answers."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return p.health()
        except RemoteError as e:
            if time.monotonic() > deadline:
                raise SystemExit(f"provider {p.name} at {p.url} does not answer: {e}")
            time.sleep(1)


def zipf_picker(names, a, rnd):
    """Objects ranked at random; object of rank k is read with weight 1 / (k + 1)^a (a = 0: uniform)."""
    order = list(names)
    rnd.shuffle(order)
    weights = [1.0 / (k + 1) ** a for k in range(len(order))]
    return lambda: rnd.choices(order, weights)[0]


def locality_picker(owner, names, fraction, rnd):
    """Nine reads in ten land in the lowest `fraction` of each type's identifier range."""
    hot, cold = [], []
    for name in names:
        obj_id = owner.catalog.get(name)[0]
        lo, hi = tag_range(owner.own, owner.type_table[name.split("-")[0]])
        (hot if id_to_int(obj_id) < lo + int((hi - lo) * fraction) else cold).append(name)
    if not hot or not cold:
        raise SystemExit("--locality leaves no hot or no cold objects; pick another fraction")
    return lambda: rnd.choice(hot) if rnd.random() < 0.9 else rnd.choice(cold)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", action="append", metavar="NAME=URL",
                    help="a provider service (default: A, B, C on 127.0.0.1:8001-8003)")
    ap.add_argument("--files", type=int, default=300, help="objects to load")
    ap.add_argument("--batch", type=int, default=50, help="objects per batch write (0: one write per object)")
    ap.add_argument("--object-bytes", type=int, default=1024)
    ap.add_argument("--reads", type=int, default=200, help="reads before the policy rounds")
    ap.add_argument("--zipf", type=float, default=1.0, help="skew of the reads (0 = uniform)")
    ap.add_argument("--locality", type=float, default=0.0,
                    help="instead of Zipf: 90%% of reads in the lowest fraction of each type's range")
    ap.add_argument("--rounds", type=int, default=3, help="policy rounds")
    ap.add_argument("--upper", type=int, default=1000)
    ap.add_argument("--lower", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sequential", action="store_true",
                    help="send to the DHT nodes one at a time instead of all at once")
    ap.add_argument("--out", default="results/net.csv")
    args = ap.parse_args()
    spec = args.provider or [f"{n}=http://127.0.0.1:{8001 + k}" for k, n in enumerate("ABC")]
    provs = {}
    for item in spec:
        name, _, url = item.partition("=")
        provs[name] = RemoteProvider(name, url)
    rnd = random.Random(args.seed)

    print("providers")
    debug = True
    for name, p in provs.items():
        h = wait_for(p)
        if h["name"] != name:
            raise SystemExit(f"{p.url} says it is provider {h['name']}, not {name}")
        debug = debug and h["debug"]
        print(f"  {name} at {p.url}: up{' (debug)' if h['debug'] else ''}")

    m = Meter(list(provs.values()))
    owner = Owner(s=8, rng=random.Random(args.seed), parallel=not args.sequential)
    pm = PolicyManager(owner, upper=args.upper, lower=args.lower)
    print(f"\nsteps (owner here, {args.files} objects of {args.object_bytes} B, N_up {args.upper}, N_low {args.lower}, "
          f"DHT nodes {'one at a time' if args.sequential else 'all at once'})")

    def setup():
        for p in provs.values():
            owner.register_provider(p)
        for ftype, tag, prov in TYPES:
            owner.add_type(ftype, tag, prov)
    m.run("Phase 1: register, open 3 trees", setup)

    names = [f"{TYPES[i % 3][0]}-{i}" for i in range(args.files)]
    files = [(n, rnd.randbytes(args.object_bytes), n.split("-")[0]) for n in names]
    if args.batch > 0:
        for at in range(0, len(files), args.batch):
            m.run(f"Phase 2: batch write, {min(args.batch, len(files) - at)} objects",
                  lambda at=at: owner.upload_many(files[at:at + args.batch]))
    else:
        m.run("Phase 2: upload one object", lambda: owner.upload(*files[len(owner.catalog)]), repeat=len(files))
    live = owner.live_rows()
    print(f"    forest: {len(live)} trees, {min(r.count for r in live)}-{max(r.count for r in live)} leaves, "
          f"{owner.hashes} owner hashes for every check so far")

    extra = [(f"extra-{i}", rnd.randbytes(args.object_bytes), TYPES[i % 3][0]) for i in range(20)]
    m.run("Phase 2: upload one object", lambda: owner.upload(*extra.pop()), repeat=20)
    pick = locality_picker(owner, names, args.locality, rnd) if args.locality else zipf_picker(names, args.zipf, rnd)
    workload = f"hot range {args.locality:g}" if args.locality else f"Zipf {args.zipf:g}"
    m.run("Phase 3: owner reads one object", lambda: owner.read(pick()), repeat=20)

    reader = Reader([owner.pk])
    plist = list(provs.values())

    def user_read():
        name = pick()
        obj_id, d = owner.reader_view(name, name.split("-")[0])
        return reader.read(obj_id, d, plist)
    m.run("Phase 3: data user reads, with D", user_read, repeat=20)
    audit = m.run("Phase 5: audit of the owner, s = 8", lambda: Auditor([owner.pk], s=8).audit_owner(owner.digest, plist))
    print(f"    rows recompute to D: {audit.digest_ok}; trees ok: {sum(t.ok for t in audit.trees)}/{len(audit.trees)}")

    m.run(f"Phase 3: {args.reads} reads, {workload}", lambda: [owner.read(pick()) for _ in range(args.reads)])
    for k in range(args.rounds):
        done = m.run(f"Phase 4: policy round {k + 1}", pm.tick)
        print(f"    decisions: {[(d.kind, d.reason) for d in done] or 'none'}")
    live = owner.live_rows()
    print(f"    forest: {len(live)} trees, {min(r.count for r in live)}-{max(r.count for r in live)} leaves")

    image = max((r for r in live if r.provider == "A"), key=lambda r: r.count)
    a, b, _ = m.run("Phase 4: split at the median", lambda: owner.split(image.tree_id))
    ab = m.run("Phase 4: merge the halves back", lambda: owner.merge(a.tree_id, b.tree_id))
    print(f"    owner hashes for both checks: 2 (one each); root unchanged: {ab.root == image.root}")
    moved = m.run("Phase 4: migrate A -> B (pipe)", lambda: owner.migrate(ab.tree_id, "B"))
    later = Auditor([owner.pk], s=8).audit_tree(moved, plist)
    print(f"    a later audit of B passes: {later.ok}; only then is A told to release")
    if later.ok:
        m.run("Phase 4: release the copy at A", lambda: owner.release(ab.tree_id, "A"))

    if debug:
        video = max((r for r in owner.live_rows() if r.provider == "B" and r.count >= 2), key=lambda r: r.count)
        victim = owner.catalog.id_at(video.lo, video.count // 2)
        name = owner.catalog.name_of(victim)
        provs["B"].corrupt(victim, 9)
        try:
            owner.read(name)
        except ReadFault as e:
            print(f"    fault injected at B: read of {name} -> {e.kind} at position {e.position}")
        rep = m.run("Phase 6: localize (rounds)", lambda: Localizer(owner).localize(video.tree_id))
        print(f"    verdict {rep.verdict}: {rep.names} after {rep.rounds} rounds, {rep.bytes_received} B")
        m.run("Phase 6: localize (one exchange)", lambda: Localizer(owner).localize(video.tree_id, True))
        m.run("Phase 6: recount of the whole forest", lambda: Localizer(owner).recount_forest())
        moved = m.run("Phase 4: failed check -> repair, migrate",
                      lambda: pm.failed_check("B", [video.tree_id], repairs={name: (b"repaired copy", "video")}))
        for row in moved:
            if Auditor([owner.pk], s=8).audit_tree(row, plist).ok:
                owner.release(row.tree_id, "B")
        print(f"    moved {len(moved)} tree(s) away from B: {[r.provider for r in moved]}")
    else:
        print("    (providers not in --debug: fault injection skipped)")

    back = m.run("recovery from the providers", lambda: Owner.recover(owner.mk, owner.sk, plist))
    print(f"    rebuilt D matches: {back.digest == owner.digest}; catalog {len(back.catalog)} entries")

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(m.rows[0]))
        w.writeheader()
        w.writerows(m.rows)
    print(f"\nwrote {args.out}")
    for p in plist:
        p.net.close()


if __name__ == "__main__":
    main()
