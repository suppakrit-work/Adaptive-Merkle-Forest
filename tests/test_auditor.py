"""Tests for amf/auditor.py -- the auditor's four scopes (III-A 2) and the data user (Phase 3)."""
import random
from dataclasses import replace

import pytest

from amf.common import crypto
from amf.auditor import Auditor, Reader, ReadRefused
from amf.owner import Owner
from tests.helpers import fill, tamper_stored_leaf, tree_of, world


def setup(n=30):
    owner, provs = world()
    files = fill(owner, n)
    return owner, provs, files, list(provs.values()), Auditor([owner.pk], s=4, rng=random.Random(2))


def test_the_auditor_holds_public_keys_only():
    owner, provs, files, plist, aud = setup()
    assert set(vars(aud)) == {"keys", "s", "rng"} and aud.keys == {owner.own: owner.pk}


def test_audit_of_one_owner_passes_a_healthy_forest():
    owner, provs, files, plist, aud = setup()
    rep = aud.audit_owner(owner.digest, plist)
    assert rep.ok and rep.digest_ok and len(rep.trees) == 3
    assert all(len(t.checked) == 4 for t in rep.trees)                # s positions per tree


def test_a_bad_ciphertext_names_the_provider_and_the_position():
    owner, provs, files, plist, aud = setup()
    prov, row = tree_of(owner, provs, "video")
    for obj_id in prov.live[row.key].ids:
        prov.corrupt(obj_id)
    rep = aud.audit_tree(row, plist)
    assert rep.provider == "B" and len(rep.bad_ciphertext) == 4      # every sampled position
    assert not rep.tree_level


def test_a_broken_path_hands_the_tree_to_phase_5():
    owner, provs, files, plist, aud = setup()
    prov, row = tree_of(owner, provs, "image")
    tamper_stored_leaf(prov, owner.own, row.tree_id, 0)
    rep = aud.audit_tree(row, plist, positions=[3, 5])
    assert rep.tree_level and rep.bad_path == [3, 5]


def test_a_withheld_row_is_caught_against_the_owners_d():
    owner, provs, files, plist, aud = setup()
    prov, row = tree_of(owner, provs, "document")
    for p in plist:
        p.dht.rows = {k: r for k, r in p.dht.rows.items() if k != row.key}   # every node hides one tree
    rep = aud.audit_owner(owner.digest, plist)
    assert not rep.digest_ok and not rep.ok and len(rep.trees) == 2


def test_audit_of_a_set_of_objects_touches_only_the_trees_that_cover_them():
    owner, provs, files, plist, aud = setup()
    ids = [owner.catalog.get("fimage-0")[0], owner.catalog.get("fimage-3")[0]]
    reps = aud.audit_ids(ids, owner.digest, plist)
    assert len(reps) == 1 and reps[0].provider == "A" and reps[0].ok


def test_audit_of_the_whole_forest_does_the_same_for_every_owner():
    w, provs = world()
    v = Owner()
    for p in provs.values():
        v.register_provider(p)
    v.add_type("image", 1, "A")
    fill(w, 9)
    v.upload("x", b"v's", "image")
    aud = Auditor([w.pk, v.pk], s=2)
    reps = aud.audit_all([w.digest, v.digest], list(provs.values()))
    assert [r.own for r in reps] == [w.own, v.own] and all(r.ok for r in reps)


# ---- the data user ---------------------------------------------------------------------

def test_a_data_user_reads_with_the_identifier_and_d_from_the_owner():
    owner, provs, files, plist, aud = setup()
    obj_id, d = owner.reader_view("fvideo-4", "video")
    ct = Reader([owner.pk]).read(obj_id, d, plist)
    assert crypto.decrypt(owner.mk, obj_id, ct) == files["fvideo-4"]   # the owner's key, not the reader's


def test_a_data_user_refuses_a_node_rolled_back_to_an_older_row():
    owner, provs, files, plist, aud = setup()
    name = "fimage-0"
    obj_id = owner.catalog.get(name)[0]
    old_rows = {k: dict(p.dht.rows) for k, p in provs.items()}
    owner.upload(name, b"new content", "image")
    for k, p in provs.items():
        p.dht.rows = old_rows[k]                                    # every node rolled back together
    with pytest.raises(KeyError):
        Reader([owner.pk]).row_for(obj_id, owner.digest, plist)    # nothing reaches the current D


def test_a_reader_with_an_old_d_is_as_fresh_as_one_honest_node():
    owner, provs, files, plist, aud = setup()
    obj_id, d_old = owner.reader_view("fimage-3", "image")
    rows_before = {k: dict(p.dht.rows) for k, p in provs.items()}
    owner.upload("fimage-3", b"rewritten", "image")
    # the reader still holds the old D and asks a node that has not seen the write first
    provs["B"].dht.rows = rows_before["B"]
    ct = Reader([owner.pk]).read(obj_id, d_old, [provs["B"], provs["A"]])
    assert crypto.decrypt(owner.mk, obj_id, ct) == b"rewritten"


def test_a_node_rolled_back_alone_does_not_hand_an_old_row_to_a_reader_with_an_old_d():
    owner, provs, files, plist, aud = setup()
    obj_id, d_old = owner.reader_view("fimage-3", "image")
    a = provs["A"].dht
    rows_before, d_before = dict(a.rows), a.digest(owner.own)
    owner.upload("fimage-3", b"rewritten", "image")
    a.rows, a.digests[owner.own] = rows_before, d_before         # A's node goes back, D and all
    row = Reader([owner.pk]).row_for(obj_id, d_old, plist)
    assert row == owner.row_for(obj_id)                         # B and C hold the newer row and D


def test_a_read_that_crosses_a_write_blames_no_one():
    owner, provs, files, plist, aud = setup()
    obj_id, d = owner.reader_view("fimage-3", "image")
    a = provs["A"]
    real = a.read

    def crossing(oid):                                           # the write lands between lookup and read
        a.read = real
        owner.upload("fimage-3", b"rewritten", "image")
        return real(oid)
    a.read = crossing
    ct = Reader([owner.pk]).read(obj_id, d, plist)
    assert crypto.decrypt(owner.mk, obj_id, ct) == b"rewritten"


def test_a_provider_naming_a_row_no_node_holds_is_refused():
    owner, provs, files, plist, aud = setup()
    obj_id, d = owner.reader_view("fimage-3", "image")
    provs["A"].corrupt(obj_id)
    real = provs["A"].read
    provs["A"].read = lambda oid: (lambda t: (t[0], t[1] + 1) + t[2:])(real(oid))   # "a newer version"
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d, plist)
    assert e.value.provider == "A" and e.value.kind == "wrong tree or version"


def test_a_data_user_sees_a_damaged_object():
    owner, provs, files, plist, aud = setup()
    obj_id, d = owner.reader_view("fdocument-2", "document")
    provs["C"].corrupt(obj_id)
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d, plist)
    assert e.value.kind == "ciphertext" and e.value.provider == "C"


# ---- answers that do not fit the question ---------------------------------------------------

def test_a_challenge_answer_must_cover_every_position_asked():
    owner, provs, files, plist, aud = setup(60)
    prov, row = tree_of(owner, provs, "video")
    for obj_id in prov.live[row.key].ids[1:]:
        del prov.objects[obj_id]                                     # only position 0 is still held
    real = prov.challenge
    for answer in (lambda own, tid, pos: [], lambda own, tid, pos: real(own, tid, [0] * len(pos))):
        prov.challenge = answer
        rep = aud.audit_tree(row, plist, positions=[3, 7, 11])
        assert not rep.ok and rep.missing == [3, 7, 11] and rep.checked == []


def test_a_lost_object_is_missing_to_an_audit_and_to_a_data_user():
    owner, provs, files, plist, aud = setup()
    prov, row = tree_of(owner, provs, "document")
    obj_id, d = owner.reader_view("fdocument-2", "document")
    del prov.objects[obj_id]
    i = owner.catalog.position(obj_id, row.lo)
    assert aud.audit_tree(row, plist, positions=[i]).missing == [i]
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d, plist)
    assert (e.value.provider, e.value.kind, e.value.position) == ("C", "missing", i)


def test_a_node_that_points_at_another_tree_shifts_no_blame():
    owner, provs, files, plist, aud = setup()
    owner.add_type("audio", 9, "A")
    owner.upload("song", b"la la", "audio")
    wrong = provs["C"].dht.route(owner.catalog.get("song")[0])    # signed and live, but not the right row
    provs["C"].dht.route = lambda oid: wrong
    obj_id, d = owner.reader_view("fimage-3", "image")
    ct = Reader([owner.pk]).read(obj_id, d, [provs["C"], provs["A"], provs["B"]])
    assert crypto.decrypt(owner.mk, obj_id, ct) == files["fimage-3"]


def test_one_forged_row_or_made_up_tree_does_not_stop_an_audit():
    owner, provs, files, plist, aud = setup()
    prov, row = tree_of(owner, provs, "image")
    c = provs["C"].dht
    forged = replace(row, version=row.version + 50)               # the signature is for another version
    real_get, real_ids = c.get, c.tree_ids
    c.get = lambda own, tid: forged if tid == row.tree_id else real_get(own, tid)
    c.tree_ids = lambda own: real_ids(own) + [bytes(16)]
    rep = aud.audit_owner(owner.digest, plist)
    assert rep.ok and len(rep.trees) == 3


def test_an_audit_of_several_objects_challenges_each_tree_once():
    owner, provs, files, plist, aud = setup()
    calls = []
    real = provs["A"].challenge
    provs["A"].challenge = lambda *a: calls.append(a) or real(*a)
    ids = [owner.catalog.get(f"fimage-{k}")[0] for k in (0, 3, 6, 9)]
    reps = aud.audit_ids(ids, owner.digest, plist)
    assert len(reps) == 1 and reps[0].ok and len(calls) == 1
