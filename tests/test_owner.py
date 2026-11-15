"""End-to-end tests for amf/owner.py -- Phases 1-3 and Algorithm 2 for writes."""
import random
from dataclasses import replace

import pytest

from amf.common.index import digest_tree, verify_digest, verify_row
from amf.common.merkle import depth_of, root_of
from amf.owner import Owner, ReadFault, VerificationError
from tests.helpers import TYPES, fill, tamper_stored_leaf, tree_of, world


# ---- Phase 1 -----------------------------------------------------------------------

def test_setup_writes_one_row_per_type_inside_the_owners_space():
    owner, provs = world()
    rows = owner.live_rows()
    assert len(rows) == 3 and all(r.own == owner.own and r.count == 0 and r.version == 1 for r in rows)
    assert all(verify_row(owner.pk, r) for r in rows)
    row = owner.row_for(owner.identifier("x", "video"))
    assert row.provider == "B"
    with pytest.raises(ValueError):
        owner.add_type("audio", 2, "A")                          # tag already in use


def test_d_is_signed_and_matches_the_live_rows_at_every_node():
    owner, provs = world()
    fill(owner, 9)
    assert verify_digest(owner.pk, owner.digest) and owner.digest.count == 3
    assert digest_tree(owner.live_rows())[0].root == owner.digest.root
    for p in provs.values():
        assert p.dht.digest(owner.own) == owner.digest


def test_d_follows_a_row_changed_in_place_with_one_path_and_is_rebuilt_otherwise():
    owner, provs = world()
    fill(owner, 12, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.split(row.tree_id)                                     # rows added and retired: rebuilt
    assert owner.digest.count == 4
    assert digest_tree(owner.live_rows())[0].root == owner.digest.root
    before = owner.digest_hashes
    owner.upload("fimage-1", b"rewritten", "image")              # one row changed in place
    assert owner.digest_hashes - before <= 1 + depth_of(owner.digest.count)
    assert digest_tree(owner.live_rows())[0].root == owner.digest.root
    owner.migrate(tree_of(owner, provs, "video")[1].tree_id, "A")  # a migration keeps its tid: in place too
    assert digest_tree(owner.live_rows())[0].root == owner.digest.root
    assert all(p.dht.digest(owner.own) == owner.digest for p in provs.values())


# ---- Phase 2: upload, rewrite, delete ------------------------------------------------

def test_upload_then_read_round_trip():
    owner, provs = world()
    files = fill(owner, 12)
    for name, data in files.items():
        assert owner.read(name) == data
    for row in owner.live_rows():
        p = provs[row.provider]
        assert p.live[row.key].root == row.root == root_of(owner.catalog.leaves_in(row.lo, row.hi))
        assert row.count == owner.catalog.count(row.lo, row.hi)
        assert all(q.dht.get(owner.own, row.tree_id) == row for q in provs.values())   # identical copies


def test_nothing_leaves_the_owner_in_the_clear():
    owner, provs = world()
    owner.upload("secret.txt", b"THE PLAINTEXT", "document")
    (obj_id, ct), = provs["C"].objects.items()
    assert b"PLAINTEXT" not in ct and b"secret" not in obj_id
    assert obj_id[:32] == owner.own and obj_id[32] == 3          # only the owner's and the type's tags show


def test_the_owner_keeps_ct_until_the_provider_acknowledges():
    owner, provs = world()
    owner.upload("a.jpg", b"v1", "image")
    assert owner.unacked == {}                                  # the tree went live, CT can go


def test_rewrite_keeps_identifier_and_count():
    owner, provs = world()
    owner.upload("a.jpg", b"v1", "image")
    obj_id, leaf1 = owner.catalog.get("a.jpg")
    row1 = owner.row_for(obj_id)
    owner.upload("a.jpg", b"v2", "image")
    obj_id2, leaf2 = owner.catalog.get("a.jpg")
    row2 = owner.row_for(obj_id)
    assert obj_id2 == obj_id and leaf2 != leaf1 and row2.count == row1.count == 1
    assert row2.version == row1.version + 1 and owner.read("a.jpg") == b"v2"


def test_delete_removes_the_leaf_and_the_object():
    owner, provs = world()
    fill(owner, 6, ftype="image")
    name = "fimage-2"
    obj_id, _ = owner.catalog.get(name)
    owner.delete(name)
    assert owner.catalog.get(name) is None and obj_id not in provs["A"].objects
    assert owner.row_for(obj_id).count == 5


def test_every_signed_root_is_one_the_owner_computed_itself():
    owner, provs = world()
    fill(owner, 30)
    for row in owner.live_rows():
        assert row.root == root_of(owner.catalog.leaves_in(row.lo, row.hi))


# ---- Algorithm 2 on writes: what a provider announces is checked before it is signed ----

def _lying_write(prov, field):
    real = prov.write

    def lie(obj_id, ct):
        tid, lo, hi, root, n = real(obj_id, ct)
        return {"root": (tid, lo, hi, b"\x07" * 32, n), "count": (tid, lo, hi, root, n + 1),
                "range": (tid, lo + 1, hi, root, n), "tree": (b"\x00" * 16, lo, hi, root, n)}[field]
    prov.write = lie


@pytest.mark.parametrize("field", ["root", "count", "range", "tree"])
def test_a_wrong_announcement_is_refused_and_nothing_changes(field):
    owner, provs = world()
    fill(owner, 5, ftype="image")
    before = dict(owner.index), owner.digest
    _lying_write(provs["A"], field)
    with pytest.raises(VerificationError):
        owner.upload("new.jpg", b"new", "image")
    assert (dict(owner.index), owner.digest) == before and owner.catalog.get("new.jpg") is None
    assert provs["A"].shadow == {}                              # the pending tree was thrown away


def test_a_provider_that_drops_a_leaf_in_the_new_tree_is_refused():
    owner, provs = world()
    fill(owner, 8, ftype="image")
    a = provs["A"]
    real = a.write

    def drop(obj_id, ct):
        tid, lo, hi, root, n = real(obj_id, ct)
        st = a.shadow[(owner.own, tid)].state
        root = root_of(st.leaves[:2] + [b"\x00" * 32] + st.leaves[3:])   # one old leaf swapped
        return tid, lo, hi, root, n
    a.write = drop
    with pytest.raises(VerificationError):
        owner.upload("new.jpg", b"new", "image")


# ---- Phase 3: reads ---------------------------------------------------------------------

def test_read_of_a_damaged_object_names_the_position_and_the_ciphertext():
    owner, provs = world()
    fill(owner, 9, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    victim = prov.live[row.key].ids[4]
    prov.corrupt(victim, at=20)
    with pytest.raises(ReadFault) as e:
        owner.read(owner.catalog.name_of(victim))
    assert (e.value.provider, e.value.position, e.value.kind) == ("B", 4, "ciphertext")


def test_read_with_rewritten_stored_hashes_is_a_path_fault():
    owner, provs = world()
    fill(owner, 9, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    tamper_stored_leaf(prov, owner.own, row.tree_id, 2)
    victim = prov.live[row.key].ids[6]
    with pytest.raises(ReadFault) as e:
        owner.read(owner.catalog.name_of(victim))
    assert e.value.kind == "path"                               # the bytes are fine, the stored path is not


def test_an_answer_from_another_version_is_not_taken():
    owner, provs = world()
    fill(owner, 4, ftype="image")
    a = provs["A"]
    real = a.read
    a.read = lambda obj_id: (lambda t: (t[0], t[1] - 1) + t[2:])(real(obj_id))
    with pytest.raises(ReadFault) as e:
        owner.read("fimage-1")
    assert e.value.kind == "wrong tree or version"


# ---- several owners ------------------------------------------------------------------------

def test_owners_share_providers_but_not_trees_or_rows():
    w, provs = world()
    v, _ = world(owner=Owner())                        # V registers with its own three providers...
    for p in provs.values():                           # ...and with W's as well
        v.register_provider(p)
    v.add_type("archive", 9, "C")
    w.upload("x", b"w", "image")
    v.upload("x", b"v", "archive")
    c = provs["C"]
    assert {r.own for r in c.dht.live_rows()} == {w.own, v.own}
    assert c.dht.digest(w.own) == w.digest and c.dht.digest(v.own) == v.digest
    assert w.read("x") == b"w" and v.read("x") == b"v"


# ---- recovery (III-A 1 b) -----------------------------------------------------------------

def test_owner_state_is_rebuilt_from_the_providers_by_whoever_holds_mk():
    owner, provs = world()
    files = fill(owner, 15)
    owner.delete("fimage-0")
    del files["fimage-0"]
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    assert back.type_table == owner.type_table and back.digest == owner.digest
    assert {r.tree_id: r for r in back.live_rows()} == {r.tree_id: r for r in owner.live_rows()}
    for name, data in files.items():
        assert back.read(name) == data
    assert back.catalog.get("fimage-0") is None


def test_recovery_takes_the_highest_version_when_one_provider_is_rolled_back():
    owner, provs = world()
    owner.upload("a.jpg", b"v1", "image")
    stale = {k: v for k, v in provs["B"].catalog_copy[owner.own].items()}
    owner.upload("a.jpg", b"v2", "image")
    provs["B"].catalog_copy[owner.own] = stale          # B hands out an older copy
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    assert back.read("a.jpg") == b"v2"



def test_recovery_uses_only_what_verifies():
    owner, provs = world()
    files = fill(owner, 15)
    _, row = tree_of(owner, provs, "image")
    c = provs["C"]
    real_get = c.dht.get
    c.dht.get = lambda own, tid: replace(row, version=row.version + 50) if tid == row.tree_id else real_get(own, tid)
    copy = c.catalog_copy[owner.own]
    obj_id = owner.catalog.get("fimage-0")[0]
    copy[obj_id] = (copy[obj_id][0] + 50, b"\x00" * 60)               # a planted entry that does not open
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    assert back.digest == owner.digest and len(back.live_rows()) == 3
    assert all(back.read(n) == d for n, d in files.items())


def test_recovery_refuses_rows_that_do_not_reach_the_latest_d():
    owner, provs = world()
    fill(owner, 9)
    _, row = tree_of(owner, provs, "video")
    for p in provs.values():
        p.dht.rows = {k: r for k, r in p.dht.rows.items() if k != row.key}   # every node lost one row
    with pytest.raises(VerificationError):
        Owner.recover(owner.mk, owner.sk, list(provs.values()))


# ---- batch writes: several objects bound for one tree, one announcement, one check ------------

def test_upload_many_builds_the_same_catalog_as_one_upload_at_a_time_and_keeps_trees_under_n_up():
    from amf.owner.policy import PolicyManager
    one, _ = world()
    many, provs = world(owner=Owner(mk=one.mk, sk=one.sk, s=4, rng=random.Random(1)))
    PolicyManager(one, upper=16, lower=4)
    PolicyManager(many, upper=16, lower=4)
    files = [(f"f{i}", b"data %d" % i, TYPES[i % 3]) for i in range(70)]
    for name, data, t in files:
        one.upload(name, data, t)
    rows = many.upload_many(files)
    assert rows and sorted((n, i) for n, i, _ in one.catalog.entries()) == \
        sorted((n, i) for n, i, _ in many.catalog.entries())                  # leaves differ: fresh IVs
    live = many.live_rows()
    assert all(r.count <= 16 for r in live) and sum(r.count for r in live) == 70
    assert all(r.root == root_of(many.catalog.leaves_in(r.lo, r.hi)) for r in live)   # every root is the owner's
    assert len(live) > 3                                                              # the full trees were split
    for name, data, _t in files:
        assert many.read(name) == data
    assert all(p.shadow == {} for p in provs.values()) and not many.unacked
    assert many.hashes < one.hashes                                                   # one check per batch, not per object


def test_upload_many_checks_the_announced_root_with_every_pending_leaf_in_place():
    owner, provs = world()
    files = [(f"g{i}", b"x%d" % i, "image") for i in range(5)]
    real = provs["A"].write_many

    def lying(items):
        items = list(items)
        tid, lo, hi, root, n = real(items[:-1] + [(items[-1][0], b"not what the owner sent")])
        return tid, lo, hi, root, n
    provs["A"].write_many = lying
    with pytest.raises(VerificationError):
        owner.upload_many(files)
    assert provs["A"].shadow == {} and len(owner.catalog) == 0 and not owner.unacked
    provs["A"].write_many = real
    owner.upload_many(files)
    assert len(owner.catalog) == 5


def test_upload_many_rewrites_and_mixes_types_in_one_call():
    owner, provs = world()
    owner.upload("a", b"old", "image")
    rows = owner.upload_many([("a", b"new", "image"), ("b", b"bb", "video"), ("c", b"cc", "image")])
    assert len(rows) == 2 and owner.read("a") == b"new" and owner.read("b") == b"bb"
    assert len(owner.catalog) == 3


def test_a_batch_that_overflows_splits_at_the_median_of_old_and_new_identifiers():
    """B1 of the second review: filling with the lowest identifiers and splitting at the tree's own
    median left one half with the whole rest of the batch, split again down to 2 leaves."""
    from amf.owner.policy import PolicyManager
    owner, _ = world()
    PolicyManager(owner, upper=1000, lower=250)
    owner.upload_many([(f"image-{i}", b"d", "image") for i in range(3000)])
    sizes = sorted(r.count for r in owner.live_rows() if r.count)
    assert sum(sizes) == 3000 and max(sizes) <= 1000 and min(sizes) >= 250     # one row per N_low at most
    assert len(sizes) <= 3000 // 250
    owner.upload_many([(f"image-{i}", b"d", "image") for i in range(3000, 6000)])
    sizes = sorted(r.count for r in owner.live_rows() if r.count)
    assert sum(sizes) == 6000 and max(sizes) <= 1000 and min(sizes) >= 250


def test_a_node_that_misses_a_batch_is_noted_and_brought_up_to_date():
    """B5 of the second review: the owner did not notice a node that refused a batch."""
    owner, provs = world()
    fill(owner, 6)
    real = provs["C"].receive_batch
    provs["C"].receive_batch = lambda rows, digest=None: False
    owner.upload("late", b"x", "image")
    assert owner.behind == {"C"}
    provs["C"].receive_batch = real
    obj_id = owner.catalog.get("late")[0]
    assert provs["C"].dht.lookup(obj_id).version < owner.row_for(obj_id).version      # C holds the old row
    assert owner.resend("C") and owner.behind == set()
    assert provs["C"].dht.lookup(obj_id) == owner.row_for(obj_id)
    provs["C"].receive_batch = lambda rows, digest=None: (_ for _ in ()).throw(TimeoutError("down"))
    owner.upload("later", b"y", "image")                                             # a node down is noted, not fatal
    assert owner.behind == {"C"}
