"""Tests for amf/localize.py -- Phase 5 and Algorithm 2."""
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


def test_a_lie_about_a_stored_value_is_caught_in_the_round_it_is_told():
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
    assert rep.verdict == "provider-lied" and "node 3" in rep.note


def test_a_refusal_is_charged_at_the_level_of_the_tree():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 2)
    prov.halves = lambda *a: (_ for _ in ()).throw(TimeoutError("no answer"))
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "no-answer" and rep.provider == "A"


def test_a_false_recomputed_value_is_seen_when_the_object_is_read():
    # Algorithm 2, line 16: "otherwise P reported a wrong rec"
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


def test_one_exchange_report_with_a_false_path_is_a_lie():
    owner, prov, row, loc = world(16)
    damage_object(prov, row, 3)
    real = prov.report_mismatches
    prov.report_mismatches = lambda own, tid: [(i, st, [b"\x00" * 32] * len(p)) for i, st, p in real(own, tid)]
    assert loc.localize(row.tree_id, one_exchange=True).verdict == "provider-lied"