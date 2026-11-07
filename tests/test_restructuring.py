"""Tests for Phase 4 and Algorithms 2 and 3 -- decomposition, composition, migration, evacuation."""
from dataclasses import replace

import pytest

from amf.common.index import id_to_int
from amf.common.merkle import left_size, node_hash, root_of
from amf.common.orders import Order, sign_order
from amf.owner import VerificationError
from tests.helpers import fill, tree_of, world


def big_image_tree(n=40):
    owner, provs = world()
    files = fill(owner, n, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    return owner, provs, files, prov, row


# ---- decomposition ---------------------------------------------------------------

def test_split_at_the_median_is_checked_with_one_hash():
    owner, provs, files, prov, row = big_image_tree(41)
    owner.hashes = 0
    a, b, gone = owner.split(row.tree_id)
    assert owner.hashes == 1                                      # Algorithm 3, line 6
    assert (a.count, b.count) == (left_size(41), 41 - left_size(41))
    assert node_hash(a.root, b.root) == row.root and gone.retired and gone.version == row.version + 1
    assert a.hi == b.lo and (a.lo, b.hi) == (row.lo, row.hi)
    for name, data in files.items():
        assert owner.read(name) == data


def test_split_elsewhere_recomputes_both_roots_from_the_catalog():
    owner, provs, files, prov, row = big_image_tree(40)
    p = id_to_int(owner.catalog.id_at(row.lo, 7))
    owner.hashes = 0
    a, b, _ = owner.split(row.tree_id, p)
    assert (a.count, b.count) == (7, 33) and owner.hashes == 6 + 32       # n - 1 for each new tree
    assert a.root == root_of(owner.catalog.leaves_in(a.lo, a.hi))


def _lie_on(prov, method, change):
    real = getattr(prov, method)
    setattr(prov, method, lambda *a, **k: change(real(*a, **k)))


@pytest.mark.parametrize("lie", ["count", "root", "range", "ids"])
def test_split_refuses_a_provider_that_lies(lie):
    owner, provs, files, prov, row = big_image_tree(30)
    change = {
        "count": lambda anns: [anns[0][:4] + (anns[0][4] + 1,), anns[1]],
        "root": lambda anns: [anns[0][:3] + (b"\x09" * 32,) + anns[0][4:], anns[1]],
        "range": lambda anns: [anns[0][:2] + (anns[0][2] + 1,) + anns[0][3:], anns[1]],
        "ids": lambda anns: [(b"\x00" * 16,) + anns[0][1:], anns[1]],
    }[lie]
    _lie_on(prov, "decompose", change)
    with pytest.raises(VerificationError):
        owner.split(row.tree_id)
    assert owner.index[row.tree_id] == row and len(owner.live_rows()) == 3
    assert prov.shadow == {}


def test_a_half_built_from_the_wrong_leaves_is_refused_off_the_median():
    owner, provs, files, prov, row = big_image_tree(30)
    p = id_to_int(owner.catalog.id_at(row.lo, 10))

    def swap(anns):
        st = prov.shadow[(owner.own, anns[1][0])].state
        fake = root_of(list(reversed(st.leaves)))                      # same leaves, wrong order
        return [anns[0], anns[1][:3] + (fake,) + anns[1][4:]]
    _lie_on(prov, "decompose", swap)
    with pytest.raises(VerificationError):
        owner.split(row.tree_id, p)


# ---- composition ------------------------------------------------------------------

def test_merge_of_two_halves_of_nearly_equal_size_is_checked_with_one_hash():
    owner, provs, files, prov, row = big_image_tree(21)
    a, b, _ = owner.split(row.tree_id)
    owner.hashes = 0
    ab = owner.merge(a.tree_id, b.tree_id)
    assert owner.hashes == 1 and ab.root == row.root and ab.count == 21     # Lemma 1 read the other way
    assert owner.index[a.tree_id].retired and owner.index[b.tree_id].retired
    for name, data in files.items():
        assert owner.read(name) == data


def test_merge_of_unequal_trees_recomputes_the_root():
    owner, provs, files, prov, row = big_image_tree(20)
    a, b, _ = owner.split(row.tree_id, id_to_int(owner.catalog.id_at(row.lo, 4)))
    owner.hashes = 0
    ab = owner.merge(a.tree_id, b.tree_id)
    assert owner.hashes == 19 and ab.root == row.root


def test_merge_needs_adjacent_ranges_at_one_provider():
    owner, provs = world()
    rows = sorted(owner.live_rows(), key=lambda r: r.lo)
    with pytest.raises(ValueError):
        owner.merge(rows[0].tree_id, rows[1].tree_id)           # different providers


def test_merge_refuses_a_provider_that_drops_a_leaf():
    owner, provs, files, prov, row = big_image_tree(20)
    a, b, _ = owner.split(row.tree_id, id_to_int(owner.catalog.id_at(row.lo, 4)))

    def drop(anns):
        st = prov.shadow[(owner.own, anns[0][0])].state
        return [anns[0][:3] + (root_of(st.leaves[:-1] + [b"\x00" * 32]),) + anns[0][4:]]
    _lie_on(prov, "compose", drop)
    with pytest.raises(VerificationError):
        owner.merge(a.tree_id, b.tree_id)
    assert not owner.index[a.tree_id].retired


# ---- migration ----------------------------------------------------------------------

def test_migration_moves_the_tree_and_the_source_keeps_its_copy_until_released():
    owner, provs, files, prov, row = big_image_tree(12)
    moved = owner.migrate(row.tree_id, "C")
    assert moved.provider == "C" and moved.tree_id == row.tree_id and moved.root == row.root
    assert row.key in provs["C"].live and row.key not in provs["A"].live
    assert row.key in provs["A"].retained and len(provs["A"].objects) == 12   # kept until released
    for name, data in files.items():
        assert owner.read(name) == data                                     # reads go to C now
    assert owner.release(row.tree_id, "A") == 12
    assert provs["A"].objects == {} and provs["A"].retained == {}


def test_destination_refuses_a_tree_that_does_not_match_the_signed_root():
    owner, provs, files, prov, row = big_image_tree(10)
    real = prov.export_tree
    prov.export_tree = lambda order: (lambda ids, cts: (ids, cts[:-1] + [b"x" * 40]))(*real(order))
    with pytest.raises(ValueError):
        owner.migrate(row.tree_id, "C")
    assert owner.index[row.tree_id].provider == "A"


def test_the_pipe_opens_only_on_the_owners_signed_order():
    owner, provs, files, prov, row = big_image_tree(5)
    order = Order("migrate", owner.own, (row.tree_id,), source="A", destination="C", version=row.version)
    with pytest.raises(PermissionError):
        prov.export_tree(order)                                             # unsigned
    with pytest.raises(PermissionError):
        provs["C"].import_from(sign_order(owner.sk, replace(order, destination="B")), prov)


def test_a_destination_that_announces_right_but_answers_a_challenge_wrong_is_refused():
    owner, provs, files, prov, row = big_image_tree(10)
    c = provs["C"]
    real = c.pending_challenge
    c.pending_challenge = lambda own, tid, pos: [(i, oid, ct[:-1] + b"\x00", st, p)
                                                 for i, oid, ct, st, p in real(own, tid, pos)]
    with pytest.raises(VerificationError):
        owner.migrate(row.tree_id, "C")
    assert row.key not in c.shadow and owner.index[row.tree_id].provider == "A"


def test_evacuate_repairs_first_then_moves_to_the_provider_holding_the_fewest_trees():
    owner, provs = world()
    files = fill(owner, 30)
    for row in list(owner.live_rows()):
        if row.provider == "B":
            owner.split(row.tree_id)                             # B now holds two trees, C one, A one
    owner.migrate(next(r.tree_id for r in owner.live_rows() if r.provider == "C"), "B")   # C empty
    a_tree = next(r for r in owner.live_rows() if r.provider == "A")
    victim = provs["A"].live[a_tree.key].ids[3]
    name = owner.catalog.name_of(victim)
    provs["A"].corrupt(victim)
    moved = owner.evacuate("A", repairs={name: (files[name], "image")})
    assert [m.provider for m in moved] == ["C"]                  # C holds the fewest of the owner's trees
    for n, data in files.items():
        assert owner.read(n) == data


def test_split_merge_migrate_keep_the_owners_books_consistent():
    owner, provs, files, prov, row = big_image_tree(50)
    a, b, _ = owner.split(row.tree_id)
    owner.migrate(b.tree_id, "B")
    a1, a2, _ = owner.split(a.tree_id)
    ab = owner.merge(a1.tree_id, a2.tree_id)
    for r in owner.live_rows():
        assert r.count == owner.catalog.count(r.lo, r.hi)
        assert r.root == root_of(owner.catalog.leaves_in(r.lo, r.hi))
        assert provs[r.provider].live[r.key].root == r.root
    assert ab.root == a.root
    for name, data in files.items():
        assert owner.read(name) == data


# ---- a destination or a source that does not play along ------------------------------------

def test_the_destination_must_answer_every_position_the_owner_asks():
    for lie in (lambda real: lambda own, tid, pos: [],
                lambda real: lambda own, tid, pos: real(own, tid, [0] * len(pos))):
        owner, provs, files, prov, row = big_image_tree(40)
        c = provs["C"]
        c.pending_challenge = lie(c.pending_challenge)
        with pytest.raises(VerificationError):
            owner.migrate(row.tree_id, "C")
        assert owner.index[row.tree_id].provider == "A" and row.key not in c.shadow


def test_a_release_keeps_the_objects_of_a_tree_that_came_back():
    owner, provs, files, prov, row = big_image_tree(12)
    owner.migrate(row.tree_id, "C")
    owner.migrate(row.tree_id, "A")                              # back before A was told to release
    assert owner.release(row.tree_id, "A") == 0
    assert all(owner.read(n) == d for n, d in files.items())


def test_a_release_keeps_the_objects_of_half_a_tree_that_came_back():
    owner, provs, files, prov, row = big_image_tree(12)
    owner.migrate(row.tree_id, "C")
    a, b, _ = owner.split(row.tree_id)
    owner.migrate(a.tree_id, "A")
    assert owner.release(row.tree_id, "A") == b.count            # only the half that stayed away
    assert all(owner.read(n) == d for n, d in files.items())


def test_an_old_migration_order_does_not_open_the_pipe_again():
    owner, provs, files, prov, row = big_image_tree(10)
    stale = sign_order(owner.sk, Order("migrate", owner.own, (row.tree_id,), source="A", destination="C",
                                       version=row.version - 1))
    with pytest.raises(PermissionError):
        prov.export_tree(stale)
    ids, cts = prov.export_tree(sign_order(owner.sk, replace(stale, version=row.version)))
    assert len(ids) == len(cts) == 10


def test_a_check_that_fails_in_any_way_leaves_nothing_pending():
    owner, provs, files, prov, row = big_image_tree(20)
    _lie_on(prov, "decompose", lambda anns: [anns[0][:3], anns[1]])     # an announcement cut short
    with pytest.raises(ValueError):
        owner.split(row.tree_id)
    assert prov.shadow == {} and owner.index[row.tree_id] == row


def test_evacuation_tries_another_destination_when_one_fails():
    owner, provs = world()
    files = fill(owner, 30)
    _, row = tree_of(owner, provs, "image")
    provs["B"].pending_challenge = lambda own, tid, pos: []          # B, first in line, shows nothing
    (moved,) = owner.evacuate("A")
    assert moved.tree_id == row.tree_id and moved.provider == "C"
    assert all(owner.read(n) == d for n, d in files.items())


def test_evacuation_stops_when_the_source_no_longer_recomputes_the_signed_root():
    owner, provs = world()
    fill(owner, 30)
    _, row = tree_of(owner, provs, "image")
    provs["A"].corrupt(provs["A"].live[row.key].ids[2])              # damaged, and not repaired first
    with pytest.raises(VerificationError, match="localize and repair"):
        owner.evacuate("A")
    assert owner.index[row.tree_id].provider == "A"



def test_an_old_split_or_merge_order_is_refused():
    owner, provs, files, prov, row = big_image_tree(20)
    old = sign_order(owner.sk, Order("split", owner.own, (row.tree_id,), (b"a" * 16, b"b" * 16),
                                     owner.median_boundary(row), "A", version=row.version - 1))
    with pytest.raises(PermissionError):
        prov.decompose(old)
    a, b, _ = owner.split(row.tree_id)
    stale = sign_order(owner.sk, Order("merge", owner.own, (a.tree_id, b.tree_id), (b"c" * 16,), source="A",
                                       version=a.version + 1))
    with pytest.raises(PermissionError):
        prov.compose(stale)
    assert prov.shadow == {}
