"""AMF prototype -- a walk through the five phases of the paper (run: python demo.py).

One owner, three providers, three file types.  Each step prints what the paper
says should happen and what the code did; the asserts make the demo fail loudly
if the code ever stops doing it.
"""
import random

from amf.common import crypto
from amf.auditor import Auditor, Reader
from amf.owner.localize import Localizer
from amf.common.merkle import depth_of, node_hash, root_of
from amf.owner import Owner, ReadFault, VerificationError
from amf.owner.policy import PolicyManager
from amf.provider import Provider


def say(*parts):
    print(" ".join(str(p) for p in parts))


def short(b: bytes) -> str:
    return b.hex()[:8]


def tag_of(row) -> int:
    return (row.lo >> 256) & 0xFF


def show_index(owner, title):
    say(f"\n  {title}  (D version {owner.digest.version}, T = {owner.digest.count} live rows)")
    say(f"  {'tree':8} {'tag':>3} {'provider':8} {'count':>5} {'ver':>3}  root")
    for r in sorted(owner.live_rows(), key=lambda r: r.lo):
        say(f"  {short(r.tree_id):8} {tag_of(r):>3} {r.provider:8} {r.count:>5} {r.version:>3}  {short(r.root)}")
    retired = [r for r in owner.index.values() if r.retired]
    if retired:
        say(f"  (+ {len(retired)} retired row(s), written once more with an empty range)")


def trees_of(owner, tag):
    return sorted((r for r in owner.live_rows() if tag_of(r) == tag), key=lambda r: r.lo)


def main():
    rng = random.Random(42)
    say("=" * 76)
    say("AMF prototype: Adaptive Merkle Forest, one owner, three providers")
    say("=" * 76)

    # ---- Phase 1 ----------------------------------------------------------------------
    owner = Owner(s=4, rng=rng)
    provs = {n: Provider(n) for n in "ABC"}
    for p in provs.values():
        owner.register_provider(p)
    owner.add_type("image", 1, "A")
    owner.add_type("video", 2, "B")
    owner.add_type("document", 3, "C")
    pm = PolicyManager(owner, upper=16, lower=4, gamma=4, mu=10, interval=3)
    say("\n[1] Phase 1: one empty tree per type tag inside the owner's identifier space.")
    say(f"    own = H(pk) = {short(owner.own)}...; every DHT node holds the whole table: "
        f"{[len(p.dht.rows) for p in provs.values()]} rows at A, B, C")
    say(f"    policy for the demo: N_up {pm.upper}, N_low {pm.lower}, gamma {pm.gamma:g}, mu {pm.mu}, "
        f"Delta_min {pm.interval} rounds")
    show_index(owner, "Distributed Index")

    # ---- Phase 2 ----------------------------------------------------------------------
    say("\n[2] Phase 2: id = own || type tag || HMAC(K_id, name), CT = IV || AES-256-GCM, leaf = H(id || CT).")
    files = {}
    owner.hashes = 0
    for i in range(20):
        files[f"photo{i}.jpg"] = (b"JPEG bytes of photo %d " % i) * 20
        owner.upload(f"photo{i}.jpg", files[f"photo{i}.jpg"], "image")
    for i in range(6):
        files[f"clip{i}.mp4"] = (b"MP4 bytes %d " % i) * 50
        owner.upload(f"clip{i}.mp4", files[f"clip{i}.mp4"], "video")
    for i in range(8):
        files[f"report{i}.pdf"] = (b"PDF bytes %d " % i) * 30
        owner.upload(f"report{i}.pdf", files[f"report{i}.pdf"], "document")
    obj_id, ct = next(iter(provs["A"].objects.items()))
    say(f"    A sees only (id, CT): id {short(obj_id)}... ({len(obj_id)} B, type tag {crypto.type_of(obj_id)}), "
        f"CT {len(ct)} B, 'JPEG' in CT: {b'JPEG' in ct}")
    say(f"    34 uploads, every announced root recomputed from the owner's catalog: {owner.hashes} hashes in all")
    for r in owner.live_rows():
        assert r.root == root_of(owner.catalog.leaves_in(r.lo, r.hi))
        assert r.count == owner.catalog.count(r.lo, r.hi)
    images = trees_of(owner, 1)
    say(f"    the size test ran before each write: the image tree was split at the 17th upload, "
        f"now {[r.count for r in images]} (none above N_up {pm.upper})")
    assert len(images) == 2 and all(r.count <= pm.upper for r in images)
    show_index(owner, "Distributed Index after 34 uploads")

    # ---- Phase 3 ----------------------------------------------------------------------
    say("\n[3] Phase 3: a read is hashed back to the signed root before it is decrypted.")
    pid = owner.catalog.get("photo3.jpg")[0]
    row = owner.row_for(pid)
    tid, version, i, stored, ct, path = provs[row.provider].read(pid)
    data = owner.read("photo3.jpg")
    say(f"    owner reads photo3.jpg: tree {short(tid)} v{version}, position {i}, path {len(path)} hashes "
        f"= {32 * len(path)} B, decrypted {len(data)} B, matches: {data == files['photo3.jpg']}")
    rid, d = owner.reader_view("clip1.mp4", "video")
    reader = Reader([owner.pk])
    ct = reader.read(rid, d, list(provs.values()))
    _row, _i, T, dpath = provs["A"].dht.row_with_path(owner.own, owner.row_for(rid).tree_id)
    say(f"    a data user gets id and D from the owner, checks its row against D with {len(dpath)} hashes "
        f"(T = {T}), then the object: {len(ct)} B of ciphertext accepted")
    old_d = owner.digest
    owner.upload("clip1.mp4", files["clip1.mp4"] + b" (edited)", "video")
    files["clip1.mp4"] += b" (edited)"
    ct = reader.read(rid, old_d, list(provs.values()))
    assert crypto.decrypt(owner.mk, rid, ct) == files["clip1.mp4"]
    say("    after a write, a reader still holding the old D asks every node and takes the newer D the owner signed:")
    say("    it reads the edited clip, as fresh as one honest node makes it")

    say("\n    An auditor holds public keys only; asked about this owner it takes the live rows from the nodes,")
    say("    recomputes D, and challenges s positions in every tree.")
    auditor = Auditor([owner.pk], s=4, rng=rng)
    audit = auditor.audit_owner(owner.digest, list(provs.values()))
    say(f"    rows recompute to D: {audit.digest_ok}; trees ok: "
        f"{[(t.provider, len(t.checked)) for t in audit.trees]}")
    assert audit.ok

    # ---- Phase 4: split ----------------------------------------------------------------
    say("\n[4] Phase 4: the documents' left half is read much more than the right one.")
    docs = trees_of(owner, 3)[0]
    left = owner.catalog.name_of(owner.catalog.id_at(docs.lo, 0))
    for _ in range(12):
        owner.read(left)
    owner.write_counts.clear()
    owner.hashes = 0
    done = pm.tick()
    a, b = trees_of(owner, 3)
    say(f"    round {pm.round_no}: {[(dd.kind, dd.reason) for dd in done]}")
    say(f"    split at the median into {a.count} + {b.count}: H(r_a || r_b) = old root: "
        f"{node_hash(a.root, b.root) == docs.root}, owner hashes for the check: {owner.hashes}")
    assert owner.hashes == 1 and owner.index[docs.tree_id].retired

    # ---- Phase 4: merge, held back by hysteresis -----------------------------------------
    say("\n    Five documents are deleted, two on the left and three on the right: 2 + 1 < N_low.")
    for name in ([owner.catalog.name_of(owner.catalog.id_at(a.lo, k)) for k in (0, 1)]
                 + [owner.catalog.name_of(owner.catalog.id_at(b.lo, k)) for k in (0, 1, 2)]):
        owner.delete(name)
        del files[name]
    for _ in range(2):
        say(f"    round {pm.round_no + 1}: {pm.tick()}  (too soon: Delta_min {pm.interval})")
    owner.hashes = 0
    done = pm.tick()
    (ab,) = trees_of(owner, 3)
    say(f"    round {pm.round_no}: {[(dd.kind, dd.reason) for dd in done]}; "
        f"merged tree has {ab.count} leaves, checked with {owner.hashes} hash")
    assert [dd.kind for dd in done] == ["merge"] and owner.hashes == 1
    assert all(owner.read(n) == data for n, data in files.items())
    show_index(owner, "Distributed Index after a split and a merge")

    # ---- Phase 6 ----------------------------------------------------------------------
    say("\n[5] Phase 6: one byte of one stored ciphertext at B changes (a disk, a bug, an intruder).")
    video = trees_of(owner, 2)[0]
    victim = owner.catalog.id_at(video.lo, 4)
    victim_name = owner.catalog.name_of(victim)
    provs["B"].corrupt(victim, at=30)
    try:
        owner.read(victim_name)
    except ReadFault as e:
        say(f"    read of {victim_name}: provider {e.provider}, tree {short(e.tree_id)}, position {e.position}, "
            f"kind '{e.kind}' (a failed read names its own position)")
    loc = Localizer(owner)
    rep = loc.localize(video.tree_id)
    say(f"    asked only 'what does the tree recompute to', the owner walks it: r' != r, {rep.rounds} rounds for "
        f"{video.count} leaves (depth {depth_of(video.count)}), {rep.owner_hashes} hashes, "
        f"{rep.bytes_received} B received, {rep.objects_downloaded} object downloaded")
    say(f"    verdict '{rep.verdict}': position {rep.positions} -> {rep.names}")
    assert rep.names == [victim_name]
    one = loc.localize(video.tree_id, one_exchange=True)
    say(f"    the same in one exchange: {one.rounds} round, positions {one.positions}, {one.owner_hashes} hashes")

    say("\n    A malicious B forges a stored value in round 2 of the walk to hide the damage.")
    real, calls = provs["B"].halves, {"n": 0}

    def lying(own, tree_id, k, lo, hi):
        calls["n"] += 1
        out = real(own, tree_id, k, lo, hi)
        if calls["n"] == 2:
            j = 0 if out[0][3] != out[0][4] else 1
            kc, l0, l1, _st, rec = out[j]
            out[j] = (kc, l0, l1, rec, rec)
        return out
    provs["B"].halves = lying
    lied = loc.localize(video.tree_id)
    provs["B"].halves = real
    say(f"    caught in that round, verdict '{lied.verdict}': {lied.note}")
    assert lied.verdict == "stored-hashes-damaged"

    # ---- Phase 4: migration after the failed check -----------------------------------------
    say("\n[6] Phase 4 after a failed check: repair the named object, then move B's trees away.")
    moved = owner.evacuate("B", repairs={victim_name: (files[victim_name], "video")})
    for m in moved:
        say(f"    tree {short(m.tree_id)} -> {m.provider} (fewest of the owner's trees), same tid, v{m.version}; "
            f"root unchanged, {owner.s} positions challenged at the destination")
    say(f"    B keeps its copy until released: retained {len(provs['B'].retained)} tree(s), "
        f"{len(provs['B'].objects)} object(s)")
    dest_audit = auditor.audit_owner(owner.digest, list(provs.values()))
    assert dest_audit.ok
    dropped = sum(owner.release(m.tree_id, "B") for m in moved)
    say(f"    a later audit passes, the owner sends the release order: B drops {dropped} object(s)")
    assert all(owner.read(n) == data for n, data in files.items())
    show_index(owner, "Distributed Index after the migration")

    # ---- Algorithm 3 refusing a lie ---------------------------------------------------------
    say("\n[7] Algorithm 3: a provider that announces a wrong root at upload is refused, nothing changes.")
    target = trees_of(owner, 1)[0].provider
    real_write = provs[target].write
    provs[target].write = lambda i, c: (lambda t: t[:3] + (bytes(32),) + t[4:])(real_write(i, c))
    before = dict(owner.index), owner.digest
    try:
        owner.upload("photo99.jpg", b"new", "image")
    except VerificationError as e:
        say(f"    VerificationError: {e}")
    provs[target].write = real_write
    say(f"    index and D unchanged: {(dict(owner.index), owner.digest) == before}; "
        f"Shadow Area at {target} empty: {not provs[target].shadow}")

    # ---- recovery -------------------------------------------------------------------------
    say("\n[8] The owner's machine is lost; whoever holds MK and the signing key rebuilds its state.")
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    say(f"    catalog {len(back.catalog)} entries, {len(back.live_rows())} live rows, D v{back.digest.version}, "
        f"types {back.type_table}")
    assert back.digest == owner.digest and len(back.catalog) == len(owner.catalog)
    say(f"    every file reads back: {all(back.read(n) == data for n, data in files.items())}")

    say("\nDone: every check passed.")


if __name__ == "__main__":
    main()
