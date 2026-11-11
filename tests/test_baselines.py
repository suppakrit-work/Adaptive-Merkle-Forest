"""Tests for the baselines of Section V: each does what its paper says, on AMF's own trees."""
import random

import pytest

from amf.common import crypto
from amf.common.merkle import MerkleTree, depth_of, root_of, verify_path
from amf.owner.policy import PolicyManager
from baselines import amf_static, mizrahi2024, wang2011, zhu2012
from tests.helpers import fill, tree_of, world


def leaves_of(n, seed=1):
    rnd = random.Random(seed)
    return [rnd.randbytes(32) for _ in range(n)]


# ---- wang2011: one tree, path proofs, the path-update check, fetch-all ------------------

def test_root_from_path_matches_the_tree_at_every_position():
    for n in (1, 2, 3, 7, 33):
        leaves = leaves_of(n)
        t = MerkleTree(leaves)
        for i in range(n):
            assert wang2011.root_from_path(n, i, leaves[i], t.path(i)) == t.root


def test_path_update_costs_two_paths_and_gives_the_new_root():
    leaves = leaves_of(33)
    t = MerkleTree(leaves)
    i, new = 20, b"\x07" * 32
    check = wang2011.PathUpdate()
    new_root = check.check(t.root, 33, i, leaves[i], new, t.path(i))
    assert check.hashes == 2 * len(t.path(i)) and len(t.path(i)) in (wang2011.depth(33) - 1, wang2011.depth(33))
    leaves[i] = new
    assert new_root == root_of(leaves)


def test_path_update_refuses_a_wrong_old_block_or_path():
    leaves = leaves_of(9)
    t = MerkleTree(leaves)
    check = wang2011.PathUpdate()
    with pytest.raises(ValueError):
        check.check(t.root, 9, 4, b"\x00" * 32, b"\x01" * 32, t.path(4))
    with pytest.raises(ValueError):
        check.check(t.root, 9, 4, leaves[4], b"\x01" * 32, t.path(4)[:-1])
    assert check.hashes == 0


def test_single_tree_costs_follow_from_n():
    d = depth_of(10000)
    c = wang2011.access_costs(10000, reads=900, writes=100)
    assert (c["owner_hashes"], c["provider_leaves"], c["read_path"]) == (200 * d, 100 * d, 900 * d)
    assert wang2011.proof_bytes(1000) == 32 * 10 + 32 and wang2011.fetch_all_bytes(1000) == 32_000


def test_fetch_all_finds_a_damaged_object():
    owner, provs = world()
    fill(owner, 12, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    prov.corrupt(owner.catalog.id_at(row.lo, 4), 3)
    found, nbytes = wang2011.fetch_all(prov, owner.own, row.tree_id, owner.catalog.leaves_in(row.lo, row.hi))
    assert found == [4] and nbytes == 32 * row.count


# ---- zhu2012: re-challenging halves --------------------------------------------------------

def test_rechallenge_finds_the_bad_block_in_about_log_n_rounds():
    for n in (1, 2, 5, 16, 333, 1000):
        for bad in {0, n // 3, n - 1}:
            calls = []

            def check_range(lo, hi):
                calls.append((lo, hi))
                return not lo <= bad < hi
            where, rounds, blocks = zhu2012.localize(check_range, n)
            d = depth_of(n)
            assert where == bad and d - 1 <= rounds <= d and rounds == len(calls)
            assert n <= blocks <= zhu2012.expected(n)[1]


def test_rechallenge_runs_on_a_provider():
    owner, provs = world()
    fill(owner, 20, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    prov.corrupt(owner.catalog.id_at(row.lo, 13), 2)

    def check_range(lo, hi):
        for i, obj_id, ct, _, path in prov.challenge(owner.own, row.tree_id, range(lo, hi)):
            if ct is None or not verify_path(row.root, row.count, i, crypto.leaf_hash(obj_id, ct), path):
                return False
        return True
    where, rounds, blocks = zhu2012.localize(check_range, row.count)
    assert where == 13 and 4 <= rounds <= 5 and row.count < blocks <= zhu2012.expected(row.count)[1]


# ---- mizrahi2024: rebuilding the roots at the verifier ------------------------------------

def test_rebuild_at_verifier_gives_the_roots_of_the_split():
    owner, provs = world()
    fill(owner, 21, ftype="document")
    prov, row = tree_of(owner, provs, "document")
    t = prov.live[row.key]
    r_a, r_b, ms, nbytes = mizrahi2024.rebuild_at_verifier(t.ids, prov.objects, 11)
    a, b, _ = owner.split(row.tree_id)                       # the median of 21 leaves is 11
    assert (r_a, r_b) == (a.root, b.root) and ms >= 0
    assert nbytes == sum(len(i) + len(prov.objects[i]) for i in t.ids) > 21 * 1024 // 8


# ---- amf_static: the forest that never restructures ----------------------------------------

def test_static_forest_never_splits_where_amf_does():
    owner, provs = world()
    assert amf_static.no_policy(owner) is None and owner.policy is None
    fill(owner, 30, ftype="image")
    assert [r.count for r in owner.live_rows() if r.count] == [30]

    adaptive, _ = world()
    PolicyManager(adaptive, upper=20, lower=5)
    fill(adaptive, 30, ftype="image")
    assert len([r for r in adaptive.live_rows() if r.count]) >= 2
    assert max(r.count for r in adaptive.live_rows()) <= 20


def test_one_tree_costs_follow_from_n():
    d = depth_of(1000)
    c = amf_static.one_tree_costs(1000, reads=10, writes=5)
    assert (c["owner_hashes"], c["provider_leaves"], c["read_path"]) == (5 * (d + 1), 5 * d, 10 * d)
