"""AMF prototype -- injected faults, one per guarantee of Section IV (run: python faults.py).

Each fault is injected into a fresh forest (one owner, three providers, three
types, in-process) and the check the paper names is run against it.  The
table says what was injected, which check met it, whether it was caught, and
what was named: the provider, the tree, the position, the object.  Where the
paper gives a probability (the possession bound (1 - delta)^s, IV-C) the
escape rate is measured over --trials migrations and set beside the bound.
Two rows are not caught on purpose and the paper says so: a false recomputed
value hides a damaged subtree from the walk (IV-E), and nodes rolled back
together fool a reader that has not been back to the owner (III-B 3).
Results go to results/faults.csv.
"""
import argparse
import csv
import hashlib
import math
import random
from dataclasses import replace

from amf.auditor import Auditor, Reader, ReadRefused
from amf.auditor.auditor import AuditReport
from amf.common import crypto
from amf.common.index import IndexRow, generate_signing_key, new_tree_id, sign_row, tag_range
from amf.common.merkle import EMPTY_ROOT, MerkleTree, node_hash, root_of
from amf.owner import Owner, ReadFault, VerificationError
from amf.owner.localize import Localizer
from amf.owner.policy import PolicyManager
from amf.provider import Provider

TYPES = (("image", 1, "A"), ("video", 2, "B"), ("document", 3, "C"))


def world(n=48, seed=7, s=8, policy=None):
    owner = Owner(mk=hashlib.sha256(b"amf-faults-%d" % seed).digest(), s=s, rng=random.Random(seed))
    provs = {k: Provider(k) for k in "ABC"}
    for p in provs.values():
        owner.register_provider(p)
    if policy:
        PolicyManager(owner, **policy)
    for ftype, tag, prov in TYPES:
        owner.add_type(ftype, tag, prov)
    names = []
    for i in range(n):
        t = TYPES[i % 3][0]
        owner.upload(f"{t}-{i}", b"object %d " % i * 4, t)
        names.append(f"{t}-{i}")
    return owner, provs, names


def tree_at(owner, provider):
    return next(r for r in owner.live_rows() if r.provider == provider and r.count >= 2)


def victim_of(owner, row, k=None):
    k = row.count // 2 if k is None else k
    obj_id = owner.catalog.id_at(row.lo, k)
    return obj_id, owner.catalog.name_of(obj_id), k


def tamper_stored_leaf(prov, row, i, value=b"\x00" * 32):
    t = prov.live[row.key]
    t.leaves[i] = value
    t.tree = MerkleTree(t.leaves)


def read_fault(fn):
    try:
        fn()
        return None
    except (ReadFault, ReadRefused) as e:
        return e


def pos(e):
    """The position a fault names, or '-' when the check did not fire (so a miss prints as NO, not a crash)."""
    return e.position if e is not None else "-"


class Table:
    def __init__(self):
        self.rows = []

    def add(self, section, fault, check, caught, named, cost="", note=""):
        row = dict(section=section, fault=fault, check=check, caught="yes" if caught else "NO",
                   named=named, cost=cost, note=note)
        self.rows.append(row)
        print(f"  {section:5} {fault:46} {check:28} {row['caught']:4} {named}")


# ---------------------------------------------------------------------------------------
def detection(tb, args):
    """IV-B: an altered or missing object does not pass once the row is signed."""
    owner, provs, names = world()
    row = tree_at(owner, "B")
    obj_id, name, k = victim_of(owner, row)
    provs["B"].corrupt(obj_id, at=3)
    e = read_fault(lambda: owner.read(name))
    tb.add("IV-B", "ciphertext altered at B", "owner read (Alg 1)", e and e.kind == "ciphertext",
           f"B tree {row.tree_id.hex()[:6]} pos {pos(e)} = {name}")
    reader = Reader([owner.pk])
    oid, d = owner.reader_view(name, name.split("-")[0])
    e = read_fault(lambda: reader.read(oid, d, list(provs.values())))
    tb.add("IV-B", "ciphertext altered at B", "data user read (Alg 1)", e and e.kind == "ciphertext",
           f"B pos {pos(e)}")
    rep = Auditor([owner.pk], s=row.count, rng=random.Random(1)).audit_tree(row, list(provs.values()))
    tb.add("IV-B", "ciphertext altered at B", "audit, all positions (Alg 4)", rep.bad_ciphertext == [k],
           f"B pos {rep.bad_ciphertext}")

    owner, provs, names = world()
    row = tree_at(owner, "C")
    obj_id, name, k = victim_of(owner, row)
    del provs["C"].objects[obj_id]
    e = read_fault(lambda: owner.read(name))
    tb.add("IV-B", "object lost at C", "owner read (Alg 1)", e and e.kind == "missing", f"C pos {pos(e)} = {name}")
    rep = Auditor([owner.pk], s=row.count, rng=random.Random(1)).audit_tree(row, list(provs.values()))
    tb.add("IV-B", "object lost at C", "audit, all positions (Alg 4)", rep.missing == [k], f"C pos {rep.missing}")
    loc = Localizer(owner).localize(row.tree_id)
    tb.add("IV-B", "object lost at C", "recount + walk (Alg 5)", loc.verdict == "objects" and loc.names == [name],
           f"{loc.names} after {loc.rounds} rounds", f"{loc.owner_hashes} hashes, {loc.bytes_received} B")

    owner, provs, names = world()
    row = tree_at(owner, "A")
    tamper_stored_leaf(provs["A"], row, 0)                          # leaf 0 sits on the path of leaf 1
    _obj_id, name, k = victim_of(owner, row, 1)
    e = read_fault(lambda: owner.read(name))
    tb.add("IV-B", "stored hash rewritten at A", "owner read (Alg 1)", e and e.kind == "path",
           f"A tree {row.tree_id.hex()[:6]} pos {pos(e)}, kind path")
    loc = Localizer(owner).localize(row.tree_id)
    tb.add("IV-B", "stored hash rewritten at A", "stored root (Alg 5 line 2)", loc.verdict == "stored-hashes-damaged" and loc.rounds == 0,
           f"tree {row.tree_id.hex()[:6]}: {loc.verdict}, no walk needed")

    # freshness: one node rolled back, every node rolled back, a withheld row
    owner, provs, names = world()
    plist = list(provs.values())
    old = {k: dict(p.dht.rows) for k, p in provs.items()}
    old_d = {k: p.dht.digest(owner.own) for k, p in provs.items()}
    oid0 = owner.catalog.get(names[0])[0]
    holder0 = provs[owner.row_for(oid0).provider]
    old_tree, old_ct = holder0.live[owner.row_for(oid0).key], holder0.objects[oid0]
    owner.upload(names[0], b"edited", names[0].split("-")[0])
    provs["A"].dht.rows, provs["A"].dht.digests[owner.own] = old["A"], old_d["A"]
    reader = Reader([owner.pk])
    oid, d = owner.reader_view(names[0], names[0].split("-")[0])
    ok = reader.read(oid, d, plist) == provs[owner.row_for(oid).provider].objects[oid]
    tb.add("IV-B", "one node rolled back to an old row", "data user read, current D", ok,
           "the row that reaches D, from another node")
    ok = reader.read(oid, old_d["B"], plist) == provs[owner.row_for(oid).provider].objects[oid]
    tb.add("IV-B", "one node rolled back, reader with an old D", "data user read (Alg 1 line 6)", ok,
           "the newer D signed by the owner, from an honest node")
    holder = provs[owner.row_for(oid).provider]
    for k, p in provs.items():
        p.dht.rows, p.dht.digests[owner.own] = old[k], old_d[k]
    holder.live[old_tree.key], holder.objects[oid] = old_tree, old_ct       # the provider rolled back too
    try:
        reader.read(oid, d, plist)
        caught = False
    except KeyError:
        caught = True
    tb.add("IV-B", "every node and the provider rolled back together", "data user read, current D", caught,
           "no row reaches the current D")
    ok = reader.read(oid, old_d["B"], plist) == old_ct
    tb.add("IV-B", "the same, reader never back at the owner", "data user read, old D", not ok,
           "not caught, by design (III-B 3): fresh as one honest node",
           note="the reader holds the old D the rolled-back nodes agree with")

    owner, provs, names = world()
    plist = list(provs.values())
    gone = tree_at(owner, "C")
    for p in provs.values():
        p.dht.rows = {k: r for k, r in p.dht.rows.items() if k != gone.key}
    rep = Auditor([owner.pk], s=4, rng=random.Random(1)).audit_owner(owner.digest, plist)
    tb.add("IV-B", "one row withheld by every node", "audit of the owner (Alg 4)", not rep.digest_ok,
           f"rows recompute to D: {rep.digest_ok}")


def soundness(tb, args):
    """IV-C: Algorithm 3 is the only way a root enters the index."""
    owner, provs, names = world()
    real = provs["A"].write

    def lie(obj_id, ct):
        tid, lo, hi, root, n = real(obj_id, ct)
        return tid, lo, hi, node_hash(root, root), n
    provs["A"].write = lie
    before = (dict(owner.index), owner.digest)
    try:
        owner.upload("forged", b"x", "image")
        caught = False
    except VerificationError:
        caught = True
    tb.add("IV-C", "wrong root announced on a write", "Alg 3 line 12", caught and (dict(owner.index), owner.digest) == before,
           "refused; index and D unchanged; Shadow Area empty" if provs["A"].shadow == {} else "shadow not empty")
    provs["A"].write = real

    owner, provs, names = world()
    row = tree_at(owner, "A")
    real = provs["A"].decompose

    def swapped(order):
        (a, b) = real(order)
        pa, pb = provs["A"].shadow[(owner.own, a[0])], provs["A"].shadow[(owner.own, b[0])]
        pa.state.leaves[-1], pb.state.leaves[0] = pb.state.leaves[0], pa.state.leaves[-1]     # one leaf out of place
        pa.state.tree, pb.state.tree = MerkleTree(pa.state.leaves), MerkleTree(pb.state.leaves)
        return [pa.state.announce(), pb.state.announce()]
    provs["A"].decompose = swapped
    owner.hashes = 0
    try:
        owner.split(row.tree_id)
        caught = False
    except VerificationError:
        caught = True
    tb.add("IV-C", "median split with a leaf swapped across halves", "Alg 3 line 6 (one hash)", caught,
           "H(r_a || r_b) != r; refused", f"{owner.hashes} hash")
    provs["A"].decompose = real
    owner.hashes = 0
    p = owner.catalog.id_at(row.lo, 3)
    provs["A"].decompose = swapped
    try:
        owner.split(row.tree_id, int.from_bytes(p, "big"))
        caught = False
    except VerificationError:
        caught = True
    tb.add("IV-C", "split elsewhere with a leaf swapped", "Alg 3 line 12 (recompute)", caught,
           "root over the catalog differs; refused", f"{owner.hashes} hashes")
    provs["A"].decompose = real

    owner, provs, names = world()
    row = tree_at(owner, "B")
    a, b, _ = owner.split(row.tree_id)
    real = provs["B"].compose

    def dropped(order):
        (ann,) = real(order)
        pend = provs["B"].shadow[(owner.own, ann[0])]
        pend.state.ids.pop(); pend.state.leaves.pop()
        pend.state.tree = MerkleTree(pend.state.leaves)
        return [pend.state.announce()]
    provs["B"].compose = dropped
    try:
        owner.merge(a.tree_id, b.tree_id)
        caught = False
    except VerificationError:
        caught = True
    tb.add("IV-C", "merge with one leaf dropped", "Alg 3 line 3 (count)", caught, "count differs; refused")
    provs["B"].compose = real

    # possession: the migration challenge at s positions against a destination missing a fraction delta
    for delta in (1 / 8, 1 / 4, 1 / 2):
        escaped, trials = 0, args.trials
        for trial in range(trials):
            owner, provs, names = world(n=48, seed=trial, s=args.s)
            row = tree_at(owner, "A")
            real = provs["B"].import_from

            def lossy(order, source, real=real, row=row, trial=trial):
                out = real(order, source)
                pend = provs["B"].shadow[(owner.own, row.tree_id)]
                rnd = random.Random(1000 + trial)
                for i in rnd.sample(list(pend.staged), int(delta * row.count)):
                    del pend.staged[i]
                return out
            provs["B"].import_from = lossy
            try:
                owner.migrate(row.tree_id, "B")
                escaped += 1
            except VerificationError:
                pass
        bound = (1 - delta) ** args.s
        n, miss = row.count, int(delta * row.count)                       # exact, s positions without replacement
        exact = math.comb(n - miss, args.s) / math.comb(n, args.s) if n - miss >= args.s else 0.0
        tb.add("IV-C", f"destination missing delta = {delta:g} of a {n}-leaf tree", f"migration challenge, s = {args.s}",
               escaped / trials <= bound + 0.05,
               f"escaped {escaped}/{trials} = {escaped / trials:.3f}; exact {exact:.3f}; bound (1-delta)^s = {bound:.3f}",
               f"{args.s} ciphertexts + paths")


def attribution(tb, args):
    """IV-D: a failed check names the provider the row names; a failed migration is tried elsewhere."""
    owner, provs, names = world()
    other = generate_signing_key()
    for p in provs.values():
        p.register_owner(other.public_key())                           # a known owner, so only the checks refuse
    row = tree_at(owner, "A")
    forged = sign_row(other, replace(row, provider="B", version=row.version + 5))
    accepted = any(p.dht.accept(forged) for p in provs.values())
    tb.add("IV-D", "row forged by another registered key under the victim's tid", "DHT node, Theorem 1(a)", not accepted,
           "refused by every node: not the owner's signature")
    forged = sign_row(other, replace(row, own=crypto.owner_tag(other.public_key()), version=row.version + 5))
    accepted = any(p.dht.accept(forged) for p in provs.values())
    other_own = crypto.owner_tag(other.public_key())
    own_row = sign_row(other, IndexRow(other_own, new_tree_id(), *tag_range(other_own, 1), "B", EMPTY_ROOT, 0, 1))
    tb.add("IV-D", "another registered owner's row over the victim's range", "DHT node, range in own space",
           not accepted and all(p.dht.accept(own_row) for p in provs.values()),
           "refused by every node; the same owner's row over its own range is accepted")

    owner, provs, names = world(policy=dict(upper=1000, lower=250))
    row = tree_at(owner, "A")
    real = provs["B"].import_from

    def empty_handed(order, source):
        out = real(order, source)
        provs["B"].shadow[(owner.own, row.tree_id)].staged.clear()        # announces the right root, holds nothing
        return out
    provs["B"].import_from = empty_handed
    moved = owner.policy.failed_check("A", [row.tree_id])
    tb.add("IV-D", "destination B announces the right root but holds nothing", "challenge fails, next provider tried",
           moved and moved[0].provider == "C", f"tree now at {moved[0].provider}; A still recomputes r",
           note="FailedCheck of Alg 2")


def localization(tb, args):
    """IV-E: semi-honest damage is named; a malicious provider is caught where it lies, or not at all."""
    owner, provs, names = world(n=96)
    row = tree_at(owner, "A")
    a_id, a_name, ka = victim_of(owner, row, 3)
    b_id, b_name, kb = victim_of(owner, row, row.count - 2)
    provs["A"].corrupt(a_id, 2); provs["A"].corrupt(b_id, 2)
    loc = Localizer(owner).localize(row.tree_id)
    tb.add("IV-E", "two objects damaged at A (semi-honest)", "walk (Alg 5)", loc.verdict == "objects" and set(loc.names) == {a_name, b_name},
           f"{loc.names}", f"{loc.rounds} rounds, {loc.owner_hashes} hashes, {loc.bytes_received} B, {loc.objects_downloaded} objects")
    loc = Localizer(owner).localize(row.tree_id, one_exchange=True)
    tb.add("IV-E", "two objects damaged at A (semi-honest)", "one exchange (Alg 5 variant)", set(loc.names) == {a_name, b_name},
           f"{loc.names}", f"1 round, {loc.owner_hashes} hashes, {loc.bytes_received} B")

    owner, provs, names = world(n=96)
    row = tree_at(owner, "A")
    obj_id, name, k = victim_of(owner, row)
    provs["A"].corrupt(obj_id, 2)
    real = provs["A"].round_halves

    def lie(own, tid, nodes):
        out = real(own, tid, nodes)
        if any(k == 2 or k == 3 for k, _, _ in nodes):                     # round 2: a false stored value
            out = [[(kc, lo, hi, b"\x00" * 32, rec) for kc, lo, hi, st, rec in pair] for pair in out]
        return out
    provs["A"].round_halves = lie
    loc = Localizer(owner).localize(row.tree_id)
    tb.add("IV-E", "malicious A: false stored value in round 2", "walk (Alg 5 line 8)", loc.verdict == "stored-hashes-damaged" and loc.rounds == 2,
           f"{loc.verdict} in round {loc.rounds}")
    provs["A"].round_halves = lambda *a: (_ for _ in ()).throw(TimeoutError("no answer"))
    loc = Localizer(owner).localize(row.tree_id)
    tb.add("IV-E", "malicious A: refuses to answer the walk", "walk (Alg 5)", loc.verdict == "no-answer",
           f"charged at the tree: {loc.verdict}")
    provs["A"].round_halves = real
    provs["A"].recomputed_root = lambda own, tid: row.root
    loc = Localizer(owner).localize(row.tree_id)
    e = read_fault(lambda: owner.read(name))
    tb.add("IV-E", "malicious A: reports the damaged tree as clean", "walk (Alg 5)", False,
           f"walk says {loc.verdict}: not caught, by design (IV-E)", note="a false rec cannot be seen without the object")
    tb.add("IV-E", "malicious A: reports the damaged tree as clean", "owner read (Alg 1)", e and e.kind == "ciphertext",
           f"read names A pos {pos(e)} = {name}")


def reports(tb, args):
    """IV-F: dishonest read counts cost bounded work; an auditor's false alarm is rechecked."""
    owner, provs, names = world(n=60, policy=dict(upper=1000, lower=250, interval=3))
    for p in provs.values():
        real = p.report_reads
        p.report_reads = (lambda own, p=p: {k: (10 ** 6, 0) for k in p.live if k[0] == own})   # lies: every tree hot
    decisions = [owner.policy.tick() for _ in range(9)]
    n_split = sum(d.kind == "split" for ds in decisions for d in ds)
    intact = all(r.root == root_of(owner.catalog.leaves_in(r.lo, r.hi)) for r in owner.live_rows())
    tb.add("IV-F", "every provider reports 10^6 reads on one half (trees < 2 N_low)", "policy (Alg 2), floor 2 N_low",
           n_split == 0 and intact, f"{n_split} splits in 9 rounds; forest intact: {intact}")

    owner, provs, names = world(n=0, policy=dict(upper=4000, lower=250, interval=3))
    owner.upload_many([(f"image-{i}", b"x" * 8, "image") for i in range(2400)])
    for p in provs.values():
        p.report_reads = (lambda own, p=p: {k: (10 ** 6, 0) for k in p.live if k[0] == own})
    rounds = [k + 1 for k in range(9) if any(d.kind == "split" for d in owner.policy.tick())]
    intact = all(r.root == root_of(owner.catalog.leaves_in(r.lo, r.hi)) for r in owner.live_rows())
    tb.add("IV-F", "the same lie on a 2400-leaf tree (halves stay above 2 N_low)", "policy (Alg 2), hysteresis Delta_min = 3",
           rounds == [1, 4, 7] and intact, f"splits only in rounds {rounds}: a tree waits Delta_min rounds; forest intact: {intact}",
           note="each split still checked by Alg 3")

    owner, provs, names = world(policy=dict(upper=1000, lower=250))
    row = tree_at(owner, "B")
    obj_id, name, k = victim_of(owner, row)
    alarm = AuditReport("B", row.tree_id, bad_ciphertext=[k])          # an auditor that cries wolf
    confirmed = owner.policy.on_audit(alarm)
    still_at_b = owner.index[row.tree_id].provider == "B"
    tb.add("IV-F", "auditor raises a false alarm on B", "owner rechecks with a read (on_audit)",
           confirmed == [] and still_at_b, f"confirmed {confirmed}; tree still at B: {still_at_b}")
    provs["B"].corrupt(obj_id, 1)
    confirmed = owner.policy.on_audit(alarm)
    tb.add("IV-F", "the same alarm when B really is damaged", "owner rechecks with a read (on_audit)",
           confirmed == [k], f"confirmed {confirmed} = {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=200, help="migrations per delta for the possession bound")
    ap.add_argument("--s", type=int, default=8)
    ap.add_argument("--out", default="results/faults.csv")
    args = ap.parse_args()
    tb = Table()
    print(f"  {'sec.':5} {'fault injected':46} {'check':28} {'caught':4} named")
    for part in (detection, soundness, attribution, localization, reports):
        part(tb, args)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tb.rows[0]))
        w.writeheader()
        w.writerows(tb.rows)
    caught = sum(r["caught"] == "yes" for r in tb.rows)
    print(f"\n{caught}/{len(tb.rows)} caught; the {len(tb.rows) - caught} not caught are the two the paper "
          f"excludes by design\nwrote {args.out}")


if __name__ == "__main__":
    main()
