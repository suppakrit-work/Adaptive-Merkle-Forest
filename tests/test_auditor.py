"""Tests for amf/auditor.py -- the auditor's four scopes (III-A 2) and the data user (Phase 3)."""
import random

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
        del p.dht.rows[row.key]                                    # every node hides one tree
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


def test_a_read_that_crosses_a_write_blames_no_one():
    owner, provs, files, plist, aud = setup()
    obj_id, d_old = owner.reader_view("fimage-3", "image")
    rows_before = {k: dict(p.dht.rows) for k, p in provs.items()}
    owner.upload("fimage-3", b"rewritten", "image")
    # the reader still holds the old D and finds a node that has not seen the write yet
    provs["B"].dht.rows = rows_before["B"]
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d_old, [provs["B"], provs["A"]])
    assert e.value.kind.startswith("newer version")
    assert Reader([owner.pk]).read(obj_id, owner.digest, plist)     # with the current D it reads fine


def test_a_data_user_sees_a_damaged_object():
    owner, provs, files, plist, aud = setup()
    obj_id, d = owner.reader_view("fdocument-2", "document")
    provs["C"].corrupt(obj_id)
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d, plist)
    assert e.value.kind == "ciphertext" and e.value.provider == "C"