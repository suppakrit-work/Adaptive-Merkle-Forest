"""AMF over a real network: the owner, a data user and an auditor against running provider services.

Start the providers first (docker compose up, or python -m amf.provider.server ...), then

    python net_demo.py                                   # the three containers on this machine
    python net_demo.py --provider A=http://1.2.3.4:8000 --provider B=... --provider C=...

The owner runs here; each provider is reached over HTTP.  Every step is timed
and the bytes and round trips it took are counted at the owner's end, so the
same script measures a run on localhost or across three clouds.  With
providers started in --debug mode it also damages one object, names it, and
moves the damaged provider's trees away.  The owner sends each new row to all
DHT nodes at once (--sequential: one after another).  Results go to results/net.csv.
"""
import argparse
import csv
import random
import statistics
import time

from amf.auditor import Auditor, Reader
from amf.owner.localize import Localizer
from amf.owner import Owner, ReadFault
from amf.provider.client import RemoteError, RemoteProvider

TYPES = (("image", 1, "A"), ("video", 2, "B"), ("document", 3, "C"))


class Meter:
    """Wall time, round trips and bytes of one step, summed over every provider the owner talks to."""

    def __init__(self, provs):
        self.provs = provs
        self.rows = []

    def counters(self):
        return [sum(getattr(p.net, k) for p in self.provs) for k in ("calls", "sent", "received")]

    def run(self, step, fn, repeat=1):
        times, out = [], None
        c0 = self.counters()
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = fn()
            times.append(1000 * (time.perf_counter() - t0))
        calls, sent, received = (b - a for a, b in zip(c0, self.counters()))
        row = {"step": step, "runs": repeat, "median_ms": round(statistics.median(times), 2),
               "calls": round(calls / repeat, 1), "sent_bytes": round(sent / repeat),
               "received_bytes": round(received / repeat)}
        self.rows.append(row)
        print(f"  {step:34} {row['median_ms']:>9.2f} ms {row['calls']:>6} calls "
              f"{row['sent_bytes']:>10,} B out {row['received_bytes']:>10,} B in")
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--provider", action="append", metavar="NAME=URL",
                    help="a provider service (default: A, B, C on 127.0.0.1:8001-8003)")
    ap.add_argument("--files", type=int, default=60)
    ap.add_argument("--object-bytes", type=int, default=1024)
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
    print(f"\nsteps (owner here, {args.files} objects of {args.object_bytes} B, "
          f"DHT nodes {'one at a time' if args.sequential else 'all at once'})")

    def setup():
        for p in provs.values():
            owner.register_provider(p)
        for ftype, tag, prov in TYPES:
            owner.add_type(ftype, tag, prov)
    m.run("Phase 1: register, open 3 trees", setup)

    names = []

    def upload_one():
        t = TYPES[len(names) % 3][0]
        name = f"{t}-{len(names)}"
        owner.upload(name, rnd.randbytes(args.object_bytes), t)
        names.append(name)
    m.run("Phase 2: upload one object", upload_one, repeat=args.files)
    m.run("Phase 3: owner reads one object", lambda: owner.read(rnd.choice(names)), repeat=20)

    reader = Reader([owner.pk])
    plist = list(provs.values())

    def user_read():
        name = rnd.choice(names)
        obj_id, d = owner.reader_view(name, name.split("-")[0])
        return reader.read(obj_id, d, plist)
    m.run("Phase 3: data user reads, with D", user_read, repeat=20)
    audit = m.run("auditor: whole owner, s = 8", lambda: Auditor([owner.pk], s=8).audit_owner(owner.digest, plist))
    print(f"    rows recompute to D: {audit.digest_ok}; trees ok: {[t.ok for t in audit.trees]}")

    image = next(r for r in owner.live_rows() if r.provider == "A")
    a, b, _ = m.run("Phase 4: split at the median", lambda: owner.split(image.tree_id))
    ab = m.run("Phase 4: merge the halves back", lambda: owner.merge(a.tree_id, b.tree_id))
    print(f"    owner hashes for both checks: 2 (one each); root unchanged: {ab.root == image.root}")
    moved = m.run("Phase 4: migrate A -> B (pipe)", lambda: owner.migrate(ab.tree_id, "B"))
    later = Auditor([owner.pk], s=8).audit_tree(moved, plist)
    print(f"    a later audit of B passes: {later.ok}; only then is A told to release")
    if later.ok:
        m.run("Phase 4: release the copy at A", lambda: owner.release(ab.tree_id, "A"))

    if debug:
        video = next(r for r in owner.live_rows() if r.provider == "B" and r.count >= 2)
        victim = owner.catalog.id_at(video.lo, video.count // 2)
        name = owner.catalog.name_of(victim)
        provs["B"].corrupt(victim, 9)
        try:
            owner.read(name)
        except ReadFault as e:
            print(f"    fault injected at B: read of {name} -> {e.kind} at position {e.position}")
        rep = m.run("Phase 5: localize (rounds)", lambda: Localizer(owner).localize(video.tree_id))
        print(f"    verdict {rep.verdict}: {rep.names} after {rep.rounds} rounds")
        m.run("Phase 5: localize (one exchange)", lambda: Localizer(owner).localize(video.tree_id, True))
        moved = m.run("evacuate B (repair, then migrate)",
                      lambda: owner.evacuate("B", repairs={name: (b"repaired copy", "video")}))
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
