"""Tests for amf/index.py -- the Distributed Index (III-A 4), III-B 2) and 3), Table I, Definition 2) and D."""
import hashlib
from dataclasses import replace

import pytest

from amf.common import crypto
from amf.common.index import (DHTNode, ForestDigest, IndexRow, digest_tree, freshest, generate_signing_key,
                       id_to_int, new_tree_id, owner_space, retire, row_leaf, sign_digest, sign_row,
                       tag_range, verify_digest, verify_row, verify_row_in_digest)
from amf.common.merkle import EMPTY_ROOT
from amf.common.orders import Order, sign_order, verify_order

SK = generate_signing_key()
PK = SK.public_key()
OWN = crypto.owner_tag(PK)
SK2 = generate_signing_key()                 # another owner
PK2 = SK2.public_key()
OWN2 = crypto.owner_tag(PK2)


def make_row(own=OWN, tag=1, provider="A", count=0, version=1):
    lo, hi = tag_range(own, tag)
    root = hashlib.sha256(b"root-%d-%d" % (tag, version)).digest()
    return IndexRow(own=own, tree_id=new_tree_id(), lo=lo, hi=hi, provider=provider, root=root,
                    count=count, version=version)


def node(*pks):
    return DHTNode(pks or [PK])


def signed_digest(sk, rows, version):
    tree, _ = digest_tree(rows)
    own = crypto.owner_tag(sk.public_key())
    return sign_digest(sk, ForestDigest(own, tree.root, tree.n, version))


# ---- the row and its signature ----------------------------------------------

def test_signed_row_verifies_and_every_field_is_covered():
    row = sign_row(SK, make_row())
    assert verify_row(PK, row) and len(row.signature) == 64
    changed = [replace(row, root=hashlib.sha256(b"other").digest()), replace(row, count=1),
               replace(row, provider="B"), replace(row, version=2), replace(row, lo=row.lo + 1),
               replace(row, hi=row.hi - 1), replace(row, tree_id=new_tree_id()), replace(row, own=OWN2)]
    for bad in changed:
        assert not verify_row(PK, bad)          # "a signature under the owner's public key over the other seven fields"


def test_only_the_owner_can_sign():
    assert not verify_row(PK, sign_row(SK2, make_row()))
    assert not verify_row(PK, make_row())        # unsigned


def test_own_is_the_hash_of_the_signing_key():
    # a row signed by owner V but naming owner W does not verify under either key
    row = sign_row(SK2, make_row(own=OWN))
    assert not verify_row(PK, row) and not verify_row(PK2, row)


def test_tree_identifiers_are_fresh():
    ids = {new_tree_id() for _ in range(1000)}
    assert len(ids) == 1000 and all(len(t) == 16 for t in ids)


# ---- ranges ------------------------------------------------------------------

def test_tag_range_holds_exactly_the_identifiers_of_that_type():
    k_id = crypto.derive_id_key(bytes(32))
    lo, hi = tag_range(OWN, 0x02)
    for name in ("a", "b", "holiday.jpg"):
        assert lo <= id_to_int(crypto.make_id(k_id, OWN, 0x02, name)) < hi
        assert not lo <= id_to_int(crypto.make_id(k_id, OWN, 0x03, name)) < hi
        assert not lo <= id_to_int(crypto.make_id(k_id, OWN2, 0x02, name)) < hi
    assert tag_range(OWN, 0x02)[1] == tag_range(OWN, 0x03)[0]        # types tile the owner's space
    assert owner_space(OWN)[0] == tag_range(OWN, 0)[0] and owner_space(OWN)[1] == tag_range(OWN, 255)[1]


def test_ranges_of_a_split_meet_exactly_at_p():
    lo, hi = tag_range(OWN, 1)
    p = lo + (hi - lo) // 2
    left, right = replace(make_row(), lo=lo, hi=p), replace(make_row(), lo=p, hi=hi)
    assert left.hi == right.lo == p
    assert left.covers(p - 1) and not left.covers(p) and right.covers(p) and not right.covers(hi)


# ---- the DHT node: signature, range, version ---------------------------------

def test_node_accepts_only_signed_rows_with_a_higher_version():
    n = node()
    v1 = sign_row(SK, make_row(version=1))
    assert n.accept(v1) and n.accept(v1)                             # the same row again changes nothing
    assert not n.accept(sign_row(SK, replace(v1, count=9)))          # another row of the same version does not
    v2 = sign_row(SK, replace(v1, version=2, count=5))
    assert n.accept(v2) and n.get(OWN, v1.tree_id).count == 5
    assert not n.accept(v1)                                          # an old row cannot come back
    assert not n.accept(replace(v2, version=3))                      # bumped version, stale signature
    assert not n.accept(sign_row(SK2, replace(v2, version=3)))


def test_node_refuses_an_owner_it_does_not_know():
    # "pk is registered with every provider and with the auditor"
    assert not node(PK).accept(sign_row(SK2, make_row(own=OWN2)))
    assert node(PK, PK2).accept(sign_row(SK2, make_row(own=OWN2)))


def test_node_refuses_a_range_outside_the_owners_identifier_space():
    n = node(PK, PK2)
    lo2, hi2 = tag_range(OWN2, 1)
    theft = sign_row(SK, replace(make_row(own=OWN), lo=lo2, hi=hi2))     # W claims V's identifiers
    assert verify_row(PK, theft) and not n.accept(theft)


def test_rows_are_keyed_by_owner_and_tree_identifier():
    # "the version is higher than the one it holds for that owner and tree identifier":
    # another owner cannot replace W's row by reusing its tree id with a higher version
    n = node(PK, PK2)
    w = sign_row(SK, make_row(own=OWN, version=1))
    v = sign_row(SK2, replace(make_row(own=OWN2), tree_id=w.tree_id, version=9))
    assert n.accept(w) and n.accept(v)
    assert n.get(OWN, w.tree_id) == w and n.get(OWN2, w.tree_id) == v


def test_retired_rows_stay_but_are_not_live():
    n = node()
    old = sign_row(SK, make_row(count=10, version=4))
    gone = sign_row(SK, retire(old))
    assert n.accept(old) and n.accept(gone)
    assert gone.retired and gone.version == 5 and n.get(OWN, old.tree_id).retired
    assert n.live_rows() == [] and not n.accept(old)


def test_a_batch_is_applied_whole_or_not_at_all():
    n = node()
    whole = sign_row(SK, make_row(version=1))
    assert n.accept(whole)
    lo, hi = whole.lo, whole.hi
    p = lo + (hi - lo) // 2
    a = sign_row(SK, replace(make_row(), lo=lo, hi=p))
    b = sign_row(SK, replace(make_row(), lo=p, hi=hi))
    bad = replace(sign_row(SK, retire(whole)), signature=bytes(64))
    assert not n.accept_batch([a, b, bad])
    assert n.live_rows() == [whole]                                 # nothing of the batch applied
    assert n.accept_batch([a, b, sign_row(SK, retire(whole))])
    assert {r.tree_id for r in n.live_rows()} == {a.tree_id, b.tree_id}


def test_lookup_routes_an_identifier_to_its_live_tree():
    n = node(PK, PK2)
    k_id = crypto.derive_id_key(bytes(32))
    w = sign_row(SK, make_row(own=OWN, tag=1))
    v = sign_row(SK2, make_row(own=OWN2, tag=1))
    assert n.accept(w) and n.accept(v)
    assert n.lookup(crypto.make_id(k_id, OWN, 1, "r.pdf")).tree_id == w.tree_id
    assert n.lookup(crypto.make_id(k_id, OWN2, 1, "r.pdf")).tree_id == v.tree_id
    assert n.lookup(crypto.make_id(k_id, OWN, 7, "no tree for this type")) is None


def test_lookup_follows_a_split_and_a_merge():
    n = node()
    k_id = crypto.derive_id_key(bytes(32))
    ids = [crypto.make_id(k_id, OWN, 1, f"f{i}") for i in range(40)]
    whole = sign_row(SK, make_row(tag=1))
    assert n.accept(whole)
    p = sorted(id_to_int(i) for i in ids)[20]
    a = sign_row(SK, replace(make_row(), lo=whole.lo, hi=p))
    b = sign_row(SK, replace(make_row(), lo=p, hi=whole.hi))
    assert n.accept_batch([a, b, sign_row(SK, retire(whole))])
    for i in ids:
        assert n.lookup(i).tree_id == (a.tree_id if id_to_int(i) < p else b.tree_id)
    ab = sign_row(SK, replace(make_row(), lo=whole.lo, hi=whole.hi))
    assert n.accept_batch([ab, sign_row(SK, retire(a)), sign_row(SK, retire(b))])
    assert all(n.lookup(i).tree_id == ab.tree_id for i in ids)


def test_live_rows_sit_in_one_b_plus_tree_keyed_by_lo():
    """III-B 2): a split deletes one key and inserts two, a merge the other way round, a migration
    only replaces the value under its key, and a retired row leaves the tree but stays in the table."""
    n = node(PK, PK2)
    k_id = crypto.derive_id_key(bytes(32))
    whole = sign_row(SK, make_row(tag=1))
    other = sign_row(SK2, make_row(own=OWN2, tag=1))
    assert n.accept_batch([whole, other])
    assert sorted(n._live) == sorted([whole.lo, other.lo]) and len(n._live) == 2
    p = id_to_int(crypto.make_id(k_id, OWN, 1, "middle"))
    a = sign_row(SK, replace(make_row(), lo=whole.lo, hi=p))
    b = sign_row(SK, replace(make_row(), lo=p, hi=whole.hi))
    assert n.accept_batch([a, b, sign_row(SK, retire(whole))])
    assert len(n._live) == 3 and n._live.get(p) == b.key and n._live.get(whole.lo) == a.key
    assert n.get(OWN, whole.tree_id).retired and len(n.rows) == 4
    moved = sign_row(SK, replace(b, provider="C", version=2))
    assert n.accept(moved) and len(n._live) == 3 and n._live.get(p) == moved.key
    ab = sign_row(SK, replace(make_row(), lo=whole.lo, hi=whole.hi))
    assert n.accept_batch([ab, sign_row(SK, retire(a)), sign_row(SK, retire(moved))])
    assert len(n._live) == 2 and n._live.get(whole.lo) == ab.key and p not in n._live
    n._live.check()


def test_a_lookup_is_a_predecessor_search_and_then_a_range_check():
    """Algorithm 1, line 3: the row with the largest lo not above the identifier, which must then
    lie below that row's hi; an identifier in a gap between two types has a predecessor but no tree."""
    n = node()
    k_id = crypto.derive_id_key(bytes(32))
    one, three = sign_row(SK, make_row(tag=1)), sign_row(SK, make_row(tag=3))
    assert n.accept_batch([one, three])
    assert n.lookup(crypto.make_id(k_id, OWN, 1, "x")).tree_id == one.tree_id
    assert n.lookup(crypto.make_id(k_id, OWN, 2, "x")) is None        # predecessor is tree 1, hi fails
    assert n.lookup(crypto.make_id(k_id, OWN, 3, "x")).tree_id == three.tree_id
    assert n.lookup(crypto.make_id(k_id, OWN, 0, "x")) is None        # below every lo


def test_the_tree_of_d_is_kept_per_owner_and_rebuilt_only_for_the_owner_that_changed():
    n = node(PK, PK2)
    mine = [sign_row(SK, make_row(tag=t)) for t in (1, 2, 3)]
    theirs = sign_row(SK2, make_row(own=OWN2, tag=1))
    assert n.accept_batch(mine, signed_digest(SK, mine, 1)) and n.accept(theirs)
    before = n._dtrees[OWN]
    assert n.accept(sign_row(SK2, replace(theirs, provider="B", version=2)))
    assert n._dtrees[OWN] is before                                   # another owner's change: untouched
    d = n.digest(OWN)
    for r in mine:
        got, i, T, path = n.row_with_path(OWN, r.tree_id)
        assert T == 3 and verify_row_in_digest(d, got, i, path)
    assert n.accept(sign_row(SK, replace(mine[0], provider="C", version=2)))
    assert n._dtrees[OWN] is not before                               # this owner's change: rebuilt


def test_many_owners_keep_their_rows_apart():
    sks = [generate_signing_key() for _ in range(30)]
    n = DHTNode([sk.public_key() for sk in sks])
    k_id = crypto.derive_id_key(bytes(32))
    rows = {}
    for sk in sks:
        own = crypto.owner_tag(sk.public_key())
        rows[own] = [sign_row(sk, make_row(own=own, tag=t)) for t in (1, 2, 3)]
        assert n.accept_batch(rows[own])
    for own, mine in rows.items():
        assert sorted(n.tree_ids(own)) == sorted(r.tree_id for r in mine)
        assert sorted(n.live_rows(own), key=lambda r: r.tree_id) == sorted(mine, key=lambda r: r.tree_id)
        for t, row in zip((1, 2, 3), mine):
            assert n.lookup(crypto.make_id(k_id, own, t, "x")) == row
        assert n.lookup(crypto.make_id(k_id, own, 9, "x")) is None
    assert len(n.rows) == 90 and n.tree_ids(bytes(32)) == []


def test_retired_trees_are_still_listed_for_their_owner():
    n = node(PK, PK2)
    old = sign_row(SK, make_row())
    other = sign_row(SK2, make_row(own=OWN2))
    assert n.accept(old) and n.accept(other) and n.accept(sign_row(SK, retire(old)))
    assert n.tree_ids(OWN) == [old.tree_id] and n.live_rows(OWN) == []
    assert n.lookup(crypto.make_id(crypto.derive_id_key(bytes(32)), OWN, 1, "x")) is None


def test_the_table_is_replaced_whole_never_edited_in_place():
    n = node()
    k_id = crypto.derive_id_key(bytes(32))
    v1 = sign_row(SK, make_row(provider="A"))
    assert n.accept(v1)
    before = dict(n.rows)
    assert n.accept(sign_row(SK, replace(v1, provider="B", version=2)))
    with pytest.raises(TypeError):
        n.rows[v1.key] = v1
    n.rows = before                                     # a node rolled back: the lookup follows
    assert n.lookup(crypto.make_id(k_id, OWN, 1, "x")).provider == "A"


def test_migration_keeps_the_tree_identifier():
    n = node()
    row = sign_row(SK, make_row(provider="A"))
    moved = sign_row(SK, replace(row, provider="C", version=2))
    assert n.accept(row) and n.accept(moved)
    assert n.get(OWN, row.tree_id).provider == "C" and len(n.rows) == 1


def test_reader_keeps_the_highest_version_across_nodes():
    a, b, c = node(), node(), node()
    v1 = sign_row(SK, make_row(version=1))
    v2 = sign_row(SK, replace(v1, version=2))
    for x in (a, b, c):
        x.accept(v1)
    b.accept(v2)
    assert freshest([x.get(OWN, v1.tree_id) for x in (a, b, c)]).version == 2


# ---- the forest digest D ---------------------------------------------------------

def test_digest_is_a_canonical_root_over_the_live_rows_in_tree_identifier_order():
    rows = [sign_row(SK, make_row(tag=t)) for t in (1, 2, 3, 4, 5)]
    tree, ordered = digest_tree(list(reversed(rows)))
    assert tree.root == digest_tree(rows)[0].root                   # order of arrival does not matter
    assert [r.tree_id for r in ordered] == sorted(r.tree_id for r in rows) and tree.n == 5
    _, live = digest_tree([sign_row(SK, retire(rows[0]))] + rows[1:])
    assert rows[0].tree_id not in [r.tree_id for r in live]          # retired rows are not leaves of D
    assert digest_tree([])[0].root == EMPTY_ROOT


def test_digest_is_signed_by_the_owner():
    rows = [sign_row(SK, make_row(tag=t)) for t in (1, 2)]
    d = signed_digest(SK, rows, 1)
    assert verify_digest(PK, d)
    assert not verify_digest(PK, replace(d, version=2)) and not verify_digest(PK2, d)


def test_a_reader_checks_its_row_against_d_with_a_short_path():
    n = node()
    rows = [sign_row(SK, make_row(tag=t)) for t in range(1, 9)]
    d = signed_digest(SK, rows, 1)
    assert n.accept_batch(rows, d) and n.digest(OWN) == d
    for r in rows:
        got, i, T, path = n.row_with_path(OWN, r.tree_id)
        assert got == r and T == 8 and len(path) == 3              # ceil(log2 T) hashes
        assert verify_row_in_digest(d, got, i, path)


def test_an_older_complete_set_of_rows_is_caught_with_the_current_d():
    # "a set of nodes rolled back together to an older but complete set of rows is caught"
    rows = [sign_row(SK, make_row(tag=t)) for t in (1, 2, 3)]
    d1 = signed_digest(SK, rows, 1)
    newer = sign_row(SK, replace(rows[0], count=7, version=2))
    d2 = signed_digest(SK, [newer] + rows[1:], 2)
    stale = node()
    assert stale.accept_batch(rows, d1)                              # this node never saw the update
    got, i, T, path = stale.row_with_path(OWN, rows[0].tree_id)
    assert verify_row_in_digest(d1, got, i, path)
    assert not verify_row_in_digest(d2, got, i, path)               # the reader holding the current D refuses it


def test_a_withheld_row_changes_the_recomputed_digest():
    rows = [sign_row(SK, make_row(tag=t)) for t in (1, 2, 3, 4)]
    d = signed_digest(SK, rows, 1)
    assert digest_tree(rows)[0].root == d.root
    assert digest_tree(rows[:3])[0].root != d.root                  # "catches a withheld one"


def test_node_keeps_only_a_newer_digest():
    n = node()
    rows = [sign_row(SK, make_row())]
    d1, d2 = signed_digest(SK, rows, 1), signed_digest(SK, rows, 2)
    assert n.accept_batch(rows, d2)
    assert not n.accept_batch([sign_row(SK, replace(rows[0], version=2))], d1)
    assert n.digest(OWN) == d2


def test_row_leaf_covers_the_whole_row():
    r = sign_row(SK, make_row())
    assert row_leaf(r) != row_leaf(replace(r, count=1)) and len(row_leaf(r)) == 32



def test_route_answers_a_reader_in_one_go():
    n = node()
    rows = [sign_row(SK, make_row(tag=t)) for t in (1, 2, 3)]
    d = signed_digest(SK, rows, 1)
    assert n.accept_batch(rows, d)
    k_id = crypto.derive_id_key(bytes(32))
    row, i, T, path, got = n.route(crypto.make_id(k_id, OWN, 2, "x"))
    assert row == rows[1] and T == 3 and got == d and verify_row_in_digest(d, row, i, path)
    assert n.route(crypto.make_id(k_id, OWN, 9, "x"))[0] is None



def test_a_malformed_row_digest_or_order_is_refused_not_a_crash():
    n = node()
    good = sign_row(SK, make_row())
    for bad in (replace(good, lo=-1), replace(good, count=2 ** 70), replace(good, provider=None),
                replace(good, signature=None)):
        assert not verify_row(PK, bad) and not n.accept(bad)
    d = signed_digest(SK, [good], 1)
    assert not verify_digest(PK, replace(d, count=-1))
    order = sign_order(SK, Order("split", OWN, (good.tree_id,), (new_tree_id(), new_tree_id()), 5, "A", version=1))
    assert not verify_order(PK, replace(order, boundary=-5))
