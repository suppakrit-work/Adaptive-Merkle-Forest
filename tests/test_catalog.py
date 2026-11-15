"""Tests for amf/catalog.py -- the owner's catalog (III-A 1 b), Table II, Phases 2, 4, 5)."""
import hashlib
import random

import pytest
from cryptography.exceptions import InvalidTag

from amf.common import crypto
from amf.owner.catalog import (TYPES_KEY, Catalog, newest, open_entry, seal_entry, seal_tombstone,
                         seal_types)
from amf.common.index import id_to_int, tag_range
from amf.common.merkle import MerkleTree, left_size, node_hash, root_of

MK = bytes(range(32))
K_ID = crypto.derive_id_key(MK)
OWN = hashlib.sha256(b"owner W").digest()


def leaf_for(name: str) -> bytes:
    return hashlib.sha256(b"leaf:" + name.encode()).digest()


def ident(name, tag=1):
    return crypto.make_id(K_ID, OWN, tag, name)


def rng(tag=1):
    return tag_range(OWN, tag)


def filled(names, tag=1) -> Catalog:
    cat = Catalog()
    for n in names:
        cat.put(n, ident(n, tag), leaf_for(n))
    return cat


# ---- names <-> identifiers <-> leaves ---------------------------------------

def test_put_get_remove_and_reverse_lookup():
    cat = filled(["a.txt", "b.txt", "c.txt"])
    obj_id, leaf = cat.get("b.txt")
    assert obj_id == ident("b.txt") and leaf == leaf_for("b.txt")
    assert cat.name_of(obj_id) == "b.txt" and cat.leaf_of(obj_id) == leaf
    cat.remove("b.txt")
    assert cat.get("b.txt") is None and cat.name_of(obj_id) is None and len(cat) == 2


def test_rewrite_keeps_the_identifier_and_replaces_the_leaf():
    cat = filled(["a.txt"])
    obj_id, _ = cat.get("a.txt")
    new_leaf = hashlib.sha256(b"rewritten").digest()
    cat.put("a.txt", obj_id, new_leaf)
    assert len(cat) == 1 and cat.get("a.txt") == (obj_id, new_leaf)
    assert cat.leaves_in(*rng()) == [new_leaf]
    with pytest.raises(ValueError):
        cat.put("a.txt", ident("other"), new_leaf)


def test_identifiers_come_out_in_identifier_order():
    names = ["f%d" % i for i in range(200)]
    cat = filled(names)
    lo, hi = rng()
    ids = cat.ids_in(lo, hi)
    assert ids == sorted(ids) and len(ids) == 200 and set(ids) == {ident(n) for n in names}
    assert cat.leaves_in(lo, hi) == [leaf_for(cat.name_of(i)) for i in ids]


# ---- the owner's own books: counts and roots of ranges -------------------------

def test_the_owner_computes_the_root_of_any_range_by_itself():
    # "the owner can compute the root of any identifier range by itself, n - 1 hash evaluations"
    cat = filled(["f%d" % i for i in range(301)])
    lo, hi = rng()
    ids = cat.ids_in(lo, hi)
    p = id_to_int(ids[150])
    n_a, n_b = cat.count(lo, p), cat.count(p, hi)
    assert (cat.count(lo, hi), n_a, n_b) == (301, 150, 151)
    r = root_of(cat.leaves_in(lo, hi))
    assert r == MerkleTree([leaf_for(cat.name_of(i)) for i in ids]).root
    # Lemma 1: at the median the two halves' roots hash to the whole
    m = id_to_int(ids[left_size(301)])
    assert node_hash(root_of(cat.leaves_in(lo, m)), root_of(cat.leaves_in(m, hi))) == r


def test_counts_match_brute_force_for_many_random_ranges():
    rnd = random.Random(3)
    cat = filled(["f%d" % i for i in range(120)])
    lo, hi = rng()
    ids = cat.ids_in(lo, hi)
    bounds = [lo] + [id_to_int(ids[i]) for i in sorted(rnd.sample(range(len(ids)), 20))] + [hi]
    for _ in range(60):
        a, b = sorted(rnd.sample(bounds, 2))
        inside = [i for i in ids if a <= id_to_int(i) < b]
        assert cat.count(a, b) == len(inside) and cat.ids_in(a, b) == inside


def test_other_types_and_owners_do_not_leak_into_a_range():
    cat = Catalog()
    for tag, name in [(1, "x"), (2, "y"), (2, "z"), (3, "w")]:
        cat.put(name, ident(name, tag), leaf_for(name))
    other = crypto.make_id(K_ID, hashlib.sha256(b"owner V").digest(), 2, "q")
    cat.put("q", other, leaf_for("q"))
    assert cat.count(*rng(2)) == 2 and cat.count(*rng(5)) == 0


# ---- positions: the chain position -> identifier -> name (Phase 6) --------------

def test_position_and_id_at_are_inverse():
    cat = filled(["f%d" % i for i in range(50)])
    lo, hi = rng()
    for k, obj_id in enumerate(cat.ids_in(lo, hi)):
        assert cat.position(obj_id, lo) == k and cat.id_at(lo, k) == obj_id
    with pytest.raises(IndexError):
        cat.id_at(lo, 50)


def test_positions_inside_the_right_half_start_at_zero():
    cat = filled(["f%d" % i for i in range(40)])
    ids = cat.ids_in(*rng())
    p = id_to_int(ids[25])
    assert cat.position(ids[25], p) == 0 and cat.position(ids[39], p) == 14 and cat.id_at(p, 0) == ids[25]


def test_an_upload_moves_later_positions_by_one():
    cat = filled(["f%d" % i for i in range(30)])
    lo, hi = rng()
    before = cat.ids_in(lo, hi)
    new_id = ident("newcomer")
    cat.put("newcomer", new_id, leaf_for("newcomer"))
    j = cat.position(new_id, lo)
    assert cat.id_at(lo, j) == new_id
    for k, obj_id in enumerate(before):
        assert cat.position(obj_id, lo) == (k if k < j else k + 1)
    cat.remove("newcomer")
    assert cat.ids_in(lo, hi) == before


def test_b_plus_tree_agrees_with_a_sorted_list_under_random_updates():
    rnd = random.Random(11)
    cat, oracle = Catalog(), {}
    lo, hi = rng()
    for step in range(600):
        name = "n%d" % rnd.randrange(150)
        if name in oracle and rnd.random() < 0.4:
            cat.remove(name)
            del oracle[name]
        else:
            obj_id = oracle[name][0] if name in oracle else ident(name)
            leaf = hashlib.sha256(b"%s-%d" % (name.encode(), step)).digest()
            cat.put(name, obj_id, leaf)
            oracle[name] = (obj_id, leaf)
        if step % 50 == 49:
            pairs = sorted(oracle.values())
            assert cat.ids_in(lo, hi) == [i for i, _ in pairs] and len(cat) == len(pairs)
            assert cat.leaves_in(lo, hi) == [l for _, l in pairs]


def test_bulk_load_equals_incremental_puts():
    names = ["f%d" % i for i in range(300)]
    entries = [(n, ident(n), leaf_for(n)) for n in names]
    fast, slow = Catalog.bulk_load(entries), filled(names)
    lo, hi = rng()
    assert fast.ids_in(lo, hi) == slow.ids_in(lo, hi) and fast.leaves_in(lo, hi) == slow.leaves_in(lo, hi)
    fast.put("extra", ident("extra"), leaf_for("extra"))
    assert len(fast) == 301
    with pytest.raises(ValueError):
        Catalog.bulk_load(entries + [entries[0]])


def test_large_catalog_counts_stay_cheap():
    cat = filled(["f%d" % i for i in range(5000)])
    lo, hi = rng()
    mid = id_to_int(cat.ids_in(lo, hi)[2500])
    assert cat.count(lo, mid) + cat.count(mid, hi) == 5000


# ---- the encrypted copy at every provider, entry by entry -------------------------

def test_each_change_sends_only_the_new_entry_encrypted_on_its_own():
    blob = seal_entry(MK, "a.jpg", ident("a.jpg"), leaf_for("a.jpg"), version=7)
    assert b"a.jpg" not in blob                                    # the provider sees ciphertext only
    assert open_entry(MK, ident("a.jpg"), 7, blob) == {"name": "a.jpg", "leaf": leaf_for("a.jpg").hex()}


def test_an_entry_cannot_be_moved_to_another_slot_or_version():
    blob = seal_entry(MK, "a.jpg", ident("a.jpg"), leaf_for("a.jpg"), version=7)
    with pytest.raises(InvalidTag):
        open_entry(MK, ident("b.jpg"), 7, blob)
    with pytest.raises(InvalidTag):
        open_entry(MK, ident("a.jpg"), 8, blob)
    with pytest.raises(InvalidTag):
        open_entry(bytes(range(1, 33)), ident("a.jpg"), 7, blob)  # wrong master key


def test_catalog_and_type_table_are_rebuilt_from_the_highest_versions():
    # "tagged with the version of D; on recovery take the highest version"
    a, b = ident("a.jpg"), ident("b.jpg")
    old = {a: (1, seal_entry(MK, "a.jpg", a, leaf_for("old"), 1)),
           TYPES_KEY: (1, seal_types(MK, {"image": 1}, 1))}
    new = {a: (3, seal_entry(MK, "a.jpg", a, leaf_for("a.jpg"), 3)),
           b: (4, seal_entry(MK, "b.jpg", b, leaf_for("b.jpg"), 4)),
           TYPES_KEY: (2, seal_types(MK, {"image": 1, "video": 2}, 2))}
    cat, types = Catalog.from_copy(MK, newest(old, new))           # a provider rolled back loses
    assert types == {"image": 1, "video": 2}
    assert cat.get("a.jpg") == (a, leaf_for("a.jpg")) and cat.get("b.jpg") == (b, leaf_for("b.jpg"))
    gone = newest(old, new, {b: (5, seal_tombstone(MK, b, 5))})
    cat2, _ = Catalog.from_copy(MK, gone)
    assert cat2.get("b.jpg") is None and len(cat2) == 1



def test_rebuilding_from_several_copies_passes_over_an_entry_that_does_not_open():
    a = ident("a.jpg")
    good = {a: (3, seal_entry(MK, "a.jpg", a, leaf_for("a.jpg"), 3)), TYPES_KEY: (1, seal_types(MK, {"image": 1}, 1))}
    planted = {a: (9, b"\x00" * 64), TYPES_KEY: (9, b"junk")}     # higher versions, but they do not open
    cat, types = Catalog.from_copies(MK, [planted, good])
    assert types == {"image": 1} and cat.get("a.jpg") == (a, leaf_for("a.jpg"))
