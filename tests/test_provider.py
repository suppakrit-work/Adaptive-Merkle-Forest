"""Tests for amf/provider.py -- the Storage Service (III-A 3), Shadow Area, Proof Engine)."""
from dataclasses import replace

import pytest

from amf.common import crypto
from amf.common.index import sign_row
from amf.common.merkle import EMPTY_ROOT, left_size, verify_path
from amf.common.orders import Order, sign_order
from amf.owner import Owner, ReadFault
from tests.helpers import fill, tree_of, world


def test_setup_opens_an_empty_tree_only_for_rows_naming_this_provider():
    owner, provs = world()
    for p in provs.values():
        assert len(p.dht.rows) == 3 and len(p.live) == 1          # every node holds the whole table
    prov, row = tree_of(owner, provs, "video")
    assert prov.name == "B" and prov.live[row.key].root == EMPTY_ROOT


def test_a_write_is_pending_until_the_signed_row_arrives():
    owner, provs = world()
    a = provs["A"]
    obj_id, ct, _ = owner.prepare("x.jpg", b"pixels", "image")
    tid, lo, hi, root, n = a.write(obj_id, ct)
    assert (owner.own, tid) in a.shadow and obj_id not in a.objects       # stored with the pending tree
    assert a.live[(owner.own, tid)].count == 0                           # the live tree keeps serving
    row = owner.index[tid]
    assert a.receive_row(sign_row(owner.sk, replace(row, root=root, count=n, version=2)))
    assert a.live[(owner.own, tid)].root == root and a.objects[obj_id] == ct
    assert a.acknowledged(owner.own, tid, root)                         # the tree went live: acknowledged


def test_a_row_with_another_root_does_not_promote_the_pending_tree():
    owner, provs = world()
    a = provs["A"]
    obj_id, ct, _ = owner.prepare("x.jpg", b"pixels", "image")
    tid, *_ = a.write(obj_id, ct)
    row = owner.index[tid]
    a.receive_row(sign_row(owner.sk, replace(row, root=b"\x01" * 32, count=1, version=2)))
    assert a.live[(owner.own, tid)].count == 0 and (owner.own, tid) in a.shadow
    assert not a.acknowledged(owner.own, tid, b"\x01" * 32)


def test_only_one_operation_pending_per_tree():
    owner, provs = world()
    a = provs["A"]
    i1, c1, _ = owner.prepare("x.jpg", b"1", "image")
    i2, c2, _ = owner.prepare("y.jpg", b"2", "image")
    a.write(i1, c1)
    with pytest.raises(RuntimeError):
        a.write(i2, c2)


def test_the_announcement_carries_tree_range_root_and_count():
    owner, provs = world()
    a = provs["A"]
    obj_id, ct, leaf = owner.prepare("x.jpg", b"pixels", "image")
    tid, lo, hi, root, n = a.write(obj_id, ct)
    row = owner.index[tid]
    assert (lo, hi, n) == (row.lo, row.hi, 1) and root == leaf        # a one-leaf tree's root is the leaf


def test_objects_go_to_the_tree_whose_range_covers_them():
    owner, provs = world()
    fill(owner, 9)
    assert {len(p.objects) for p in provs.values()} == {3}
    for p in provs.values():
        (t,) = p.live.values()
        assert all(t.lo <= int.from_bytes(i, "big") < t.hi for i in t.ids) and t.ids == sorted(t.ids)


def test_read_answers_with_version_position_stored_leaf_ciphertext_and_path():
    owner, provs = world()
    fill(owner, 12)
    name = "fimage-3"
    obj_id, leaf = owner.catalog.get(name)
    row = owner.row_for(obj_id)
    tid, version, i, stored, ct, path = provs[row.provider].read(obj_id)
    assert (tid, version) == (row.tree_id, row.version) and stored == leaf
    assert i == owner.catalog.position(obj_id, row.lo)
    assert verify_path(row.root, row.count, i, crypto.leaf_hash(obj_id, ct), path)


def test_challenge_answers_carry_the_identifier_and_the_stored_leaf():
    owner, provs = world()
    fill(owner, 12)
    prov, row = tree_of(owner, provs, "document")
    for i, obj_id, ct, stored, path in prov.challenge(owner.own, row.tree_id, range(row.count)):
        assert obj_id == owner.catalog.id_at(row.lo, i) and stored == crypto.leaf_hash(obj_id, ct)
        assert verify_path(row.root, row.count, i, stored, path)


def test_reads_are_reported_per_half_of_each_tree_the_owners_included():
    owner, provs = world()
    fill(owner, 12, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    names = [owner.catalog.name_of(i) for i in owner.catalog.ids_in(row.lo, row.hi)]
    for name in names[:3]:
        owner.read(name)                                   # three reads on the left half
    owner.read(names[-1])                                  # one on the right half
    assert prov.report_reads(owner.own) == {row.key: (3, 1)}
    assert prov.report_reads(owner.own) == {}              # "since its last report"


def test_a_provider_holds_trees_of_many_owners_side_by_side():
    w, provs = world()
    v = Owner()
    for p in provs.values():
        v.register_provider(p)
    v.add_type("image", 1, "A")
    w.upload("same-name.jpg", b"W's", "image")
    v.upload("same-name.jpg", b"V's", "image")
    a = provs["A"]
    assert len(a.live) == 2 and len(a.objects) == 2
    assert w.read("same-name.jpg") == b"W's" and v.read("same-name.jpg") == b"V's"
    assert {t.own for t in a.live.values()} == {w.own, v.own}


def test_orders_must_be_signed_by_the_owner_they_name():
    owner, provs = world()
    fill(owner, 6, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    bad = Order("split", owner.own, (row.tree_id,), (b"a" * 16, b"b" * 16), owner.median_boundary(row), "A",
                version=row.version)
    with pytest.raises(PermissionError):
        prov.decompose(bad)                                             # unsigned
    with pytest.raises(PermissionError):
        prov.decompose(sign_order(Owner().sk, bad))                     # signed by someone else
    with pytest.raises(PermissionError):
        prov.decompose(sign_order(owner.sk, replace(bad, source="B")))  # addressed to another provider


def test_report_mismatches_names_positions_with_their_stored_paths():
    owner, provs = world()
    fill(owner, 15, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    victim = prov.live[row.key].ids[5]
    prov.corrupt(victim)
    prov.recomputed_root(owner.own, row.tree_id)
    (got,) = prov.report_mismatches(owner.own, row.tree_id)
    i, stored, path = got
    assert i == 5 and verify_path(row.root, row.count, i, stored, path)


def test_left_half_is_the_first_ceil_n_over_2_leaves():
    owner, provs = world()
    fill(owner, 7, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    ids = prov.live[row.key].ids
    for i in range(7):
        owner.read(owner.catalog.name_of(ids[i]))
    assert prov.report_reads(owner.own)[row.key] == (left_size(7), 7 - left_size(7))



def test_a_read_names_the_version_its_live_tree_went_live_under():
    owner, provs = world()
    fill(owner, 6, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    real = prov.write

    def write_then_lose(obj_id, ct):
        announced = real(obj_id, ct)
        prov.shadow.pop((owner.own, row.tree_id))                 # the pending tree is lost before the row
        return announced
    prov.write = write_then_lose
    owner.upload("new.jpg", b"new", "image")
    with pytest.raises(ReadFault) as e:
        owner.read("fimage-0")
    assert e.value.kind == "wrong tree or version"                # not a path fault under the newer row
