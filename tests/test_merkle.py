"""Tests for amf/merkle.py -- the ordered Merkle tree (paper III-B.2, III-C)."""
import hashlib
import os

import pytest

from amf.common import merkle
from amf.common.merkle import MerkleTree, verify_path, node_hash, depth_of, left_size, sides_of


def leaves_for(n: int) -> list[bytes]:
    """n distinct 32-byte leaf hashes, deterministic so tests are repeatable."""
    return [hashlib.sha256(b"leaf-%d" % i).digest() for i in range(n)]


# ---- shape rule ----------------------------------------------------------

def test_depth_is_ceil_log2():
    assert [depth_of(n) for n in (0, 1, 2, 3, 4, 5, 7, 8, 9)] == [0, 0, 1, 2, 2, 3, 3, 3, 4]
    # "a tree of 500,000 leaves takes nineteen rounds" (III-C)
    assert depth_of(500_000) == 19


def test_left_half_is_the_larger_half():
    assert [left_size(m) for m in (2, 3, 4, 5, 6, 7)] == [1, 2, 2, 3, 3, 4]


def test_sides_follow_from_n_and_i():
    assert sides_of(1, 0) == []
    assert sides_of(2, 0) == ["L"] and sides_of(2, 1) == ["R"]
    assert sides_of(3, 0) == ["L", "L"]
    assert sides_of(3, 1) == ["L", "R"]
    assert sides_of(3, 2) == ["R"]            # the right half of 3 is a single leaf
    with pytest.raises(ValueError):
        sides_of(3, 3)


# ---- roots ---------------------------------------------------------------

def test_empty_and_single_leaf_roots():
    assert MerkleTree([]).root == merkle.EMPTY_ROOT
    leaf = leaves_for(1)[0]
    assert MerkleTree([leaf]).root == leaf
    assert MerkleTree([leaf]).path(0) == []
    assert verify_path(leaf, 1, 0, leaf, [])


def test_root_is_deterministic_and_order_sensitive():
    # "keeps its leaves in identifier order": swap two leaves and the root changes
    ls = leaves_for(6)
    assert MerkleTree(ls).root == MerkleTree(list(ls)).root
    swapped = ls[:]
    swapped[1], swapped[4] = swapped[4], swapped[1]
    assert MerkleTree(swapped).root != MerkleTree(ls).root


def test_internal_node_uses_the_prefix():
    a, b = leaves_for(2)
    assert MerkleTree([a, b]).root == hashlib.sha256(b"\x01" + a + b).digest()
    assert MerkleTree([a, b]).root != hashlib.sha256(a + b).digest()


# ---- the two halves (III-C, and the O(1) split refinement for D3) --------

@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 16, 17, 100, 101])
def test_children_are_the_roots_of_the_two_halves(n):
    ls = leaves_for(n)
    tree = MerkleTree(ls)
    left_root, right_root = tree.children()
    m = left_size(n)
    assert left_root == MerkleTree(ls[:m]).root       # the left half, as a tree of its own
    assert right_root == MerkleTree(ls[m:]).root      # the right half likewise
    assert tree.root == node_hash(left_root, right_root)


# ---- paths ---------------------------------------------------------------

@pytest.mark.parametrize("n", list(range(1, 21)) + [63, 64, 65, 1000])
def test_every_position_has_a_verifying_path(n):
    ls = leaves_for(n)
    tree = MerkleTree(ls)
    for i in range(n):
        p = tree.path(i)
        assert len(p) == len(sides_of(n, i))
        assert tree.depth - 1 <= len(p) <= tree.depth or n == 1
        assert tree.leaf(i) == ls[i]
        assert verify_path(tree.root, n, i, ls[i], p)


def test_full_tree_paths_have_exactly_log2_n_siblings():
    for n in (2, 4, 8, 16, 1024):
        tree = MerkleTree(leaves_for(n))
        assert all(len(tree.path(i)) == depth_of(n) for i in range(n))


def test_path_rejects_wrong_leaf_root_position_or_sibling():
    n = 10
    ls = leaves_for(n)
    tree = MerkleTree(ls)
    i, p = 6, tree.path(6)
    other = hashlib.sha256(b"not this leaf").digest()
    assert not verify_path(tree.root, n, i, other, p)                # changed leaf
    assert not verify_path(other, n, i, ls[i], p)                    # wrong root
    assert not verify_path(tree.root, n, 7, ls[i], p)                # "the old path having to
    assert not verify_path(tree.root, n, 5, ls[i], p)                #  land at the position"
    tampered = p[:]
    tampered[0] = other
    assert not verify_path(tree.root, n, i, ls[i], tampered)         # tampered sibling
    assert not verify_path(tree.root, n + 1, i, ls[i], p)            # wrong count
    assert not verify_path(tree.root, n, i, ls[i], p[:-1])           # path too short
    assert not verify_path(tree.root, n, n, ls[i], p)                # position out of range


def test_insertion_moves_later_positions_by_one():
    # "sampled leaves shown under both roots, those past an insertion ...
    #  moved by one, and an added leaf shown under the new root at the
    #  position its identifier takes" (III-B.3)
    ls = leaves_for(9)
    old = MerkleTree(ls)
    new_leaf = hashlib.sha256(b"the new object").digest()
    j = 4
    new_ls = ls[:j] + [new_leaf] + ls[j:]
    new = MerkleTree(new_ls)
    assert new.n == old.n + 1
    assert verify_path(new.root, new.n, j, new_leaf, new.path(j))
    for i in range(old.n):
        i_new = i if i < j else i + 1
        assert verify_path(old.root, old.n, i, ls[i], old.path(i))
        assert verify_path(new.root, new.n, i_new, ls[i], new.path(i_new))


@pytest.mark.parametrize("n", [1, 2, 3, 7, 8, 13])
def test_a_leaf_changed_in_place_rehashes_only_its_path(n):
    # "a row changed in place costs ceil(log2 T) hashes" (III-A 4)
    ls = leaves_for(n)
    tree = MerkleTree(ls)
    for i in range(n):
        ls[i] = hashlib.sha256(b"changed-%d" % i).digest()
        spent = tree.update(i, ls[i])
        assert tree.root == MerkleTree(ls).root
        assert spent == len(sides_of(n, i)) <= depth_of(n)
        assert all(verify_path(tree.root, n, j, ls[j], tree.path(j)) for j in range(n))


def test_builds_a_large_tree_quickly():
    n = 1 << 15                                   # 32,768 leaves
    ls = [os.urandom(32) for _ in range(n)]
    tree = MerkleTree(ls)
    assert tree.depth == 15
    for i in (0, 1, n // 2, n - 1):
        assert verify_path(tree.root, n, i, ls[i], tree.path(i))