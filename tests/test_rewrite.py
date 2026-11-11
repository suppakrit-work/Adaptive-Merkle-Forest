"""A rewrite (an object stored again under its name) leaves the tree's shape alone, so both ends
rehash one path: the provider in its Shadow Area, the owner against the nodes it keeps."""
import pytest

from amf.common import crypto
from amf.common.merkle import MerkleTree, depth_of, root_of
from amf.owner import VerificationError
from tests.helpers import fill, tree_of, world


def test_provider_rewrites_along_the_path_and_matches_a_full_rebuild():
    owner, provs = world()
    files = fill(owner, 33, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    t = prov.live[row.key]
    j = 17
    obj_id, new_ct = t.ids[j], b"another ciphertext"
    new = t.rewritten(j, crypto.leaf_hash(obj_id, new_ct))
    assert new.ids is t.ids and new.count == 33
    assert new.root == root_of(new.leaves) == MerkleTree(new.leaves).root
    assert t.root == row.root                              # the live tree is untouched until the owner signs


def test_owner_checks_a_rewrite_with_the_path_after_building_the_nodes_once():
    owner, provs = world()
    files = fill(owner, 40, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    name = "fvideo-7"
    owner.hashes = 0
    owner.upload(name, b"new content", "video")           # first rewrite: the nodes are built (n - 1) ...
    d = depth_of(40)
    assert 39 + d - 1 <= owner.hashes <= 39 + d           # ... then one path
    owner.hashes = 0
    owner.upload(name, b"newer content", "video")         # second rewrite: the path only
    assert d - 1 <= owner.hashes <= d
    assert owner.read(name) == b"newer content"
    row = owner.index[row.tree_id]
    assert owner._trees[row.tree_id].root == row.root == root_of(owner.catalog.leaves_in(row.lo, row.hi))


def test_a_new_object_still_costs_the_whole_tree_and_drops_the_kept_nodes():
    owner, provs = world()
    fill(owner, 20, ftype="document")
    prov, row = tree_of(owner, provs, "document")
    owner.upload("fdocument-3", b"x", "document")         # builds and keeps the nodes
    assert row.tree_id in owner._trees
    owner.hashes = 0
    owner.upload("brand-new", b"y", "document")           # an insert: n - 1 hashes, the kept nodes go
    assert owner.hashes == 20 and row.tree_id not in owner._trees
    owner.upload("fdocument-3", b"z", "document")         # rebuilt once more, then kept
    assert owner._trees[row.tree_id].n == 21


def test_kept_nodes_go_when_the_tree_is_split_and_a_wrong_rewrite_root_is_refused():
    owner, provs = world()
    fill(owner, 24, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.upload("fimage-5", b"v2", "image")
    assert row.tree_id in owner._trees
    a, b, _ = owner.split(row.tree_id)
    assert row.tree_id not in owner._trees and a.tree_id not in owner._trees
    owner.upload("fimage-5", b"v3", "image")               # lands in one half, checked along its path
    assert any(tid in owner._trees for tid in (a.tree_id, b.tree_id))

    # the provider announces a rewrite whose root is wrong: refused, nothing changes
    bad = prov.write
    def lying(obj_id, ct):
        tid, lo, hi, root, n = bad(obj_id, ct)
        return tid, lo, hi, bytes(32), n
    prov.write = lying
    before = {r.tree_id: r.version for r in owner.live_rows()}
    with pytest.raises(VerificationError):
        owner.upload("fimage-5", b"v4", "image")
    assert {r.tree_id: r.version for r in owner.live_rows()} == before and not prov.shadow
    assert owner.read("fimage-5") == b"v3"
