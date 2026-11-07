"""Tests for amf/localize.py -- Phase 6 and Algorithm 5."""
import random

import pytest

from amf.owner.localize import Localizer
from amf.common.merkle import sides_of
from amf.owner import Owner, ReadFault
from amf.provider import Provider
from tests.helpers import tamper_stored_leaf


def world(n=64, seed=5):
    owner = Owner(s=3, rng=random.Random(seed))
    provs = {k: Provider(k) for k in "AB"}
    for p in provs.values():
        owner.register_provider(p)
    owner.add_type("image", 1, "A")
    for i in range(n):
        owner.upload(f"img{i}", b"image number %d" % i, "image")
    (row,) = owner.live_rows()
    return owner, provs["A"], row, Localizer(owner)


def damage_object(prov, row, position):
    obj_id = prov.live[row.key].ids[position]
    prov.corrupt(obj_id, at=-5)
    return obj_id


def test_clean_tree_needs_no_rounds():
    owner, prov, row, loc = world(16)
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "clean" and rep.rounds == 0 and loc.recount(row.tree_id) == []


@pytest.mark.parametrize("n", [2, 3, 7, 16, 64, 100, 257])
def test_one_damaged_object_is_named_after_at_most_ceil_log2_n_rounds(n):
    owner, prov, row, loc = world(n)
    victim = 1 if n < 4 else n * 2 // 3
    obj_id = damage_object(prov, row, victim)
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "objects" and rep.positions == [victim]
    assert rep.names == [owner.catalog.name_of(obj_id)]
    assert rep.rounds == len(sides_of(n, victim)) <= (n - 1).bit_length()     # the depth of that leaf
    assert rep.owner_hashes == rep.rounds + 1                                   # one per node followed, one to settle
    assert rep.objects_downloaded == 1                                          # only the named object


def test_the_answer_is_a_name_not_a_yes_or_no():
    owner, prov, row, loc = world(40)
    obj_id = damage_object(prov, row, 17)
    rep = loc.localize(row.tree_id)
    assert rep.names == [owner.catalog.name_of(obj_id)] and rep.names[0].startswith("img")


def test_two_damaged_objects_are_both_found_and_the_walk_follows_both_halves():
    owner, prov, row, loc = world(64)
    a, b = damage_object(prov, row, 3), damage_object(prov, row, 60)
    rep = loc.localize(row.tree_id)
    assert rep.positions == [3, 60] and rep.rounds == 6
    assert set(rep.names) == {owner.catalog.name_of(a), owner.catalog.name_of(b)}


def test_one_exchange_against_a_provider_that_follows_the_protocol():
    owner, prov, row, loc = world(64)
    damage_object(prov, row, 41)
    rep = loc.localize(row.tree_id, one_exchange=True)
    assert rep.verdict == "objects" and rep.rounds == 1 and rep.positions == [41]
    assert rep.owner_hashes == 6 + 1                                            # ceil(log2 n) per position, plus the read


def test_stored_hashes_rewritten_stop_the_search_at_the_tree():
    owner, prov, row, loc = world(32)
    tamper_stored_leaf(prov, owner.own, row.tree_id, 9)
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "stored-hashes-damaged" and rep.rounds == 0


def test_a_false_stored_value_is_caught_in_the_round_it_is_told():
    owner, prov, row, loc = world(32)
    damage_object(prov, row, 20)
    real = prov.halves

    def lie(own, tid, k, lo, hi):
        out = real(own, tid, k, lo, hi)
        if k == 3:                                              # round 2, right side
            (kl, l0, l1, st, rec), right = out
            out = [(kl, l0, l1, b"\x00" * 32, rec), right]
        return out
    prov.halves = lie
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "stored-hashes-damaged" and "node 3" in rep.note       # Algorithm 5, line 8


def test_a_refusal_is_charged_at_the_level_of_the_tree():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 2)
    prov.halves = lambda *a: (_ for _ in ()).throw(TimeoutError("no answer"))
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "no-answer" and rep.provider == "A"


def test_a_false_recomputed_value_is_seen_when_the_object_is_read():
    # Algorithm 5, line 16: "otherwise P reported a wrong rec"
    owner, prov, row, loc = world(16)
    real = prov.recomputed_leaves

    def fake(own, tid):
        leaves = real(own, tid)
        leaves[5] = b"\x01" * 32
        return leaves
    prov.recomputed_leaves = fake
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "wrong-rec" and rep.wrong_rec == [5] and rep.names == []


def test_a_provider_that_reports_damage_as_clean_hides_it_from_the_walk_but_not_from_a_read():
    owner, prov, row, loc = world(16)
    obj_id = damage_object(prov, row, 7)
    prov.recomputed_root = lambda own, tid: row.root            # "a malicious one can answer from its stored hashes"
    assert loc.localize(row.tree_id).verdict == "clean"
    with pytest.raises(ReadFault) as e:
        owner.read(owner.catalog.name_of(obj_id))
    assert e.value.kind == "ciphertext" and e.value.position == 7


def test_recount_compares_every_recomputed_leaf_with_the_catalog():
    owner, prov, row, loc = world(30)
    damage_object(prov, row, 4)
    damage_object(prov, row, 22)
    assert loc.recount(row.tree_id) == [4, 22]


def test_one_exchange_report_with_a_false_path_stops_at_the_tree():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 3)
    real = prov.report_mismatches
    prov.report_mismatches = lambda own, tid: [(i, st, [b"\x00" * 32] * len(p)) for i, st, p in real(own, tid)]
    assert loc.localize(row.tree_id, one_exchange=True).verdict == "stored-hashes-damaged"


# ---- answers that do not fit the walk, and the whole forest ----------------------------------

def test_the_owner_computes_the_ranges_of_the_halves_itself():
    owner = Owner(s=3, rng=random.Random(5))
    prov = Provider("A")
    owner.register_provider(prov)
    owner.add_type("image", 1, "A")
    owner.add_type("video", 2, "A")
    for i in range(4):
        owner.upload(f"img{i}", b"image %d" % i, "image")
    for i in range(8):
        owner.upload(f"vid{i}", b"video %d" % i, "video")
    img, vid = sorted(owner.live_rows(), key=lambda r: r.lo)
    prov.corrupt(owner.catalog.id_at(vid.lo, 0))                # damage in the OTHER tree
    t = prov.live[img.key].tree

    def shifted(own, tid, k, lo, hi):                           # genuine stored values, false ranges
        if k == 1:
            return [(2, 0, 5, t.node(2), b"\x01" * 32), (3, 5, 4, t.node(3), t.node(3))]
        return [(4, 0, 4, t.node(4), t.node(4)), (5, 4, 5, t.node(5), b"\x02" * 32)]
    prov.halves = shifted
    prov.recomputed_root = lambda own, tid: b"\x05" * 32
    rep = Localizer(owner).localize(img.tree_id)
    assert rep.verdict == "provider-lied" and rep.names == [] and "node 1" in rep.note


def test_a_lost_object_is_named_by_the_walk_the_exchange_the_recount_and_a_read():
    owner, prov, row, loc = world(32)
    obj_id = prov.live[row.key].ids[9]
    del prov.objects[obj_id]
    name = owner.catalog.name_of(obj_id)
    for one in (False, True):
        rep = loc.localize(row.tree_id, one_exchange=one)
        assert rep.verdict == "objects" and rep.positions == [9] and rep.names == [name]
    assert loc.recount(row.tree_id) == [9]
    with pytest.raises(ReadFault) as e:
        owner.read(name)
    assert (e.value.kind, e.value.position) == ("missing", 9)


def test_damaged_stored_hashes_below_the_root_stop_the_search_at_the_tree():
    owner, prov, row, loc = world(32)
    damage_object(prov, row, 3)
    prov.live[row.key].tree._set(2, bytes(32))                   # a stored node under the root, root intact
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "stored-hashes-damaged" and "node 1" in rep.note


def test_a_tree_of_one_leaf_needs_no_walk():
    owner, prov, row, loc = world(1)
    damage_object(prov, row, 0)
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "objects" and rep.positions == [0] and rep.rounds == 0


def test_a_refused_read_of_the_named_object_is_charged_at_the_tree():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 4)
    prov.read = lambda obj_id: (_ for _ in ()).throw(TimeoutError("no answer"))
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "no-answer" and rep.provider == "A" and rep.positions == [4]


def test_the_bytes_of_a_localization_include_the_object_read():
    owner, prov, row, loc = world(64)
    obj_id = damage_object(prov, row, 41)
    rep = loc.localize(row.tree_id)
    assert rep.bytes_received >= 64 + 128 * rep.rounds + len(prov.objects[obj_id])


def test_one_round_is_one_exchange_even_when_both_halves_disagree():
    owner, prov, row, loc = world(64)
    damage_object(prov, row, 3)
    damage_object(prov, row, 60)
    calls = []
    real = prov.round_halves
    prov.round_halves = lambda *a: calls.append(a) or real(*a)
    rep = loc.localize(row.tree_id)
    assert rep.positions == [3, 60] and len(calls) == rep.rounds == 6


def test_a_position_reported_twice_in_one_exchange_is_a_lie():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 3)
    real = prov.report_mismatches
    prov.report_mismatches = lambda own, tid: real(own, tid) * 3
    assert loc.localize(row.tree_id, one_exchange=True).verdict == "provider-lied"


def test_the_whole_forest_is_recounted_with_one_message_per_provider():
    owner = Owner(s=3, rng=random.Random(5))
    provs = {k: Provider(k) for k in "AB"}
    for p in provs.values():
        owner.register_provider(p)
    for tag, (ftype, prov) in enumerate([("image", "A"), ("video", "B"), ("text", "B")], start=1):
        owner.add_type(ftype, tag, prov)
        for i in range(12):
            owner.upload(f"{ftype}{i}", b"%s %d" % (ftype.encode(), i), ftype)
    provs["A"].corrupt(owner.catalog.get("image4")[0], at=-3)
    provs["B"].corrupt(owner.catalog.get("video7")[0], at=-3)
    calls = []
    for p in provs.values():
        p.recount = (lambda real: lambda own: calls.append(own) or real(own))(p.recount)
    reps = Localizer(owner).localize_all()
    assert sorted(n for r in reps for n in r.names) == ["image4", "video7"] and len(calls) == 2
