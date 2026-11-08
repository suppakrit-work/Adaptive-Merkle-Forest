"""A malicious provider or DHT node may answer with anything at all.  Every verifier must then give a
verdict that names it, or pass over its answer, never crash: Algorithm 1 line 13, Algorithm 4 lines
12-14, Theorem 1(d), Phase 2 'nothing is changed'.  Each case here was an attack that once worked."""
import random
from dataclasses import replace

import pytest

from amf.auditor import Auditor, Reader, ReadRefused
from amf.common.orders import Order, sign_order
from amf.owner import Owner, ReadFault, VerificationError
from amf.owner.localize import Localizer
from amf.provider import Provider
from tests.helpers import TYPES, fill, tree_of, world


def plist(provs):
    return list(provs.values())


# ---- one bad DHT node (A1) ----------------------------------------------------------------

@pytest.mark.parametrize("bad_digest", [
    lambda d: replace(d, version="99"),                 # a version that is not a number
    lambda d: 42,                                       # not a digest at all
    lambda d: replace(d, version=10 ** 9),              # a high version with a wrong signature
])
def test_a_node_with_a_bad_digest_does_not_stop_a_data_user(bad_digest):
    owner, provs = world()
    fill(owner, 6)
    real = provs["B"].dht.route
    provs["B"].dht.route = lambda obj_id: (lambda r: (*r[:4], bad_digest(r[4])))(real(obj_id))
    obj_id, d = owner.reader_view("fimage-0", "image")
    assert Reader([owner.pk]).read(obj_id, d, plist(provs))          # the honest nodes carry the read


@pytest.mark.parametrize("bad_answer", [
    lambda a: (a[0], "0", a[2], a[3], a[4]),            # position as a string
    lambda a: (a[0], None, a[2], 7, a[4]),              # no position, a path that is an int
    lambda a: (a[0], a[1], a[2]),                       # three fields
    lambda a: None,                                     # nothing
    lambda a: (7, 0, 1, [], None),                      # a row that is not a row
])
def test_a_node_with_a_malformed_route_answer_is_passed_over(bad_answer):
    owner, provs = world()
    fill(owner, 6)
    real = provs["A"].dht.route
    provs["A"].dht.route = lambda obj_id: bad_answer(real(obj_id))
    obj_id, d = owner.reader_view("fimage-0", "image")
    assert Reader([owner.pk]).read(obj_id, d, plist(provs))
    rep = Auditor([owner.pk], s=2).audit_ids([obj_id], d, plist(provs))
    assert rep and all(r.ok for r in rep)


def test_a_node_that_raises_is_treated_as_silent():
    owner, provs = world()
    fill(owner, 6)
    provs["C"].dht.route = lambda obj_id: (_ for _ in ()).throw(TimeoutError("down"))
    provs["C"].dht.get = lambda own, tid: (_ for _ in ()).throw(TimeoutError("down"))
    provs["C"].dht.tree_ids = lambda own: (_ for _ in ()).throw(TimeoutError("down"))
    obj_id, d = owner.reader_view("fimage-0", "image")
    assert Reader([owner.pk]).read(obj_id, d, plist(provs))
    rep = Auditor([owner.pk], s=2).audit_owner(d, plist(provs))
    assert rep.digest_ok and all(t.ok for t in rep.trees)
    back = Owner.recover(owner.mk, owner.sk, plist(provs))
    assert back.digest == owner.digest


# ---- ill-typed read and challenge answers (A2) -----------------------------------------------

BAD_READS = [
    lambda a: a[:4],                                                   # four fields
    lambda a: (*a, "extra"),                                           # seven fields
    lambda a: (a[0], a[1], a[2], a[3], 12345, a[5]),                   # CT an int
    lambda a: (a[0], a[1], a[2], a[3], [1, 2], a[5]),                  # CT a list
    lambda a: (a[0], a[1], a[2], a[3], a[4], None),                    # no path
    lambda a: (a[0], a[1], a[2], a[3], a[4], [1] * len(a[5])),         # path of ints
    lambda a: (a[0], a[1], "3", a[3], a[4], a[5]),                     # position a string
]


@pytest.mark.parametrize("bad", BAD_READS)
def test_the_owner_and_a_data_user_get_a_verdict_not_a_crash_on_a_malformed_read(bad):
    owner, provs = world()
    fill(owner, 6)
    real = provs["A"].read
    provs["A"].read = lambda obj_id: bad(real(obj_id))
    with pytest.raises(ReadFault) as e:
        owner.read("fimage-0")
    assert e.value.provider == "A" and e.value.kind in ("path", "no answer", "wrong position", "missing",
                                                         "wrong tree or version")
    obj_id, d = owner.reader_view("fimage-0", "image")
    with pytest.raises(ReadRefused) as e:
        Reader([owner.pk]).read(obj_id, d, plist(provs))
    assert e.value.provider == "A"


def test_a_provider_that_raises_on_a_read_is_named():
    owner, provs = world()
    fill(owner, 6)
    provs["A"].read = lambda obj_id: (_ for _ in ()).throw(TimeoutError("no answer"))
    with pytest.raises(ReadFault) as e:
        owner.read("fimage-0")
    assert e.value.provider == "A" and e.value.kind == "no answer"


BAD_CHALLENGES = [
    lambda a: (a[0], b"\x00" + a[1], a[2], a[3], a[4]),                # a 66-byte identifier, same integer
    lambda a: (a[0], a[1], 5, a[3], a[4]),                             # CT an int
    lambda a: (a[0], a[1], a[2], a[3], [1] * len(a[4])),               # path of ints
    lambda a: ([a[0]], a[1], a[2], a[3], a[4]),                        # position a list
    lambda a: (a[0], a[1]),                                            # two fields
    lambda a: "nonsense",
]


@pytest.mark.parametrize("bad", BAD_CHALLENGES)
def test_an_audit_gives_a_verdict_on_a_malformed_challenge_answer(bad):
    owner, provs = world()
    fill(owner, 9)
    prov, row = tree_of(owner, provs, "image")
    real = prov.challenge
    prov.challenge = lambda own, tid, pos: [bad(a) for a in real(own, tid, pos)]
    rep = Auditor([owner.pk], s=3, rng=random.Random(1)).audit_tree(row, plist(provs))
    assert not rep.ok and rep.provider == "A"
    assert len(rep.bad_path) + len(rep.missing) == 3
    whole = Auditor([owner.pk], s=3, rng=random.Random(1)).audit_owner(owner.digest, plist(provs))
    assert whole.digest_ok and sum(not t.ok for t in whole.trees) == 1       # the other trees still audited


@pytest.mark.parametrize("fault", ["raise", "short", "gone"])
def test_a_provider_that_does_not_answer_a_challenge_is_missing_at_every_position(fault):
    owner, provs = world()
    fill(owner, 9)
    prov, row = tree_of(owner, provs, "image")
    if fault == "raise":
        prov.challenge = lambda own, tid, pos: (_ for _ in ()).throw(TimeoutError("down"))
    elif fault == "short":
        prov.challenge = lambda own, tid, pos: []
    else:
        del prov.live[row.key]                                          # the tree is simply gone
    rep = Auditor([owner.pk], s=3, rng=random.Random(1)).audit_tree(row, plist(provs))
    assert rep.missing and len(rep.missing) == 3 and not rep.ok


def test_a_node_handing_out_a_row_with_a_string_version_does_not_stop_an_audit_or_recovery():
    owner, provs = world()
    fill(owner, 6)
    real = provs["B"].dht.get
    provs["B"].dht.get = lambda own, tid: (lambda r: replace(r, version="7") if r else r)(real(own, tid))
    rep = Auditor([owner.pk], s=2).audit_owner(owner.digest, plist(provs))
    assert rep.digest_ok and all(t.ok for t in rep.trees)
    back = Owner.recover(owner.mk, owner.sk, plist(provs))
    assert back.digest == owner.digest and len(back.catalog) == 6


def test_a_catalog_copy_with_a_string_version_is_passed_over_at_recovery():
    owner, provs = world()
    fill(owner, 6)
    slot = owner.catalog.get("fimage-0")[0]                            # one object's entry, at every copy
    for p in provs.values():
        v, blob = p.catalog_copy[owner.own][slot]
        p.catalog_copy[owner.own][slot] = ("9", blob)
    with pytest.raises(VerificationError):                              # the copy no longer recomputes: A5
        Owner.recover(owner.mk, owner.sk, plist(provs))


def test_recount_answers_of_the_wrong_shape_are_no_answer():
    owner, provs = world()
    fill(owner, 9)
    provs["A"].recount = lambda own: [1, 2, 3]
    provs["B"].recount = lambda own: {next(iter(provs["B"].live))[1]: (b"x", b"y", b"z")}
    reports = Localizer(owner).localize_all()
    assert all(r.verdict in ("clean", "no-answer") for r in reports)


# ---- an announcement that is not well formed (A3) ----------------------------------------------

@pytest.mark.parametrize("twist", [
    lambda a: (a[0], a[1], a[2], a[3], float(a[4])),                   # count 3.0
    lambda a: (a[0], a[1], a[2], a[3], True),                          # count True
    lambda a: (a[0], a[1], a[2], a[3].hex(), a[4]),                    # root as text
    lambda a: (a[0], a[1], a[2], a[3][:31], a[4]),                     # root of 31 bytes
    lambda a: a[:4],
    lambda a: None,
])
def test_a_malformed_write_announcement_changes_nothing(twist):
    owner, provs = world()
    fill(owner, 6)
    before = (dict(owner.index), owner.digest, owner.catalog.entries())
    real = provs["A"].write
    provs["A"].write = lambda i, c: twist(real(i, c))
    with pytest.raises(VerificationError):
        owner.upload("new", b"x", "image")
    assert (dict(owner.index), owner.digest, owner.catalog.entries()) == before
    assert provs["A"].shadow == {} and not owner.unacked
    provs["A"].write = real
    owner.upload("new", b"x", "image")                                 # the tree is not wedged
    assert owner.read("new") == b"x"


def test_a_malformed_batch_announcement_changes_nothing():
    owner, provs = world()
    real = provs["A"].write_many
    provs["A"].write_many = lambda items: (lambda a: (a[0], a[1], a[2], a[3], float(a[4])))(real(items))
    with pytest.raises(VerificationError):
        owner.upload_many([(f"g{i}", b"x", "image") for i in range(4)])
    assert len(owner.catalog) == 0 and provs["A"].shadow == {} and not owner.unacked


# ---- a replayed release order (A6) ----------------------------------------------------------------

def test_a_release_order_for_an_earlier_migration_does_not_open_the_retained_copy():
    owner, provs = world()
    fill(owner, 30)
    _, row = tree_of(owner, provs, "image")
    first = owner.migrate(row.tree_id, "B")                             # A -> B, A retains
    stale = sign_order(owner.sk, Order("release", owner.own, (row.tree_id,), source="A", version=first.version))
    owner.release(row.tree_id, "A")
    owner.migrate(row.tree_id, "A")                                     # B -> A
    owner.migrate(row.tree_id, "B")                                     # A -> B again: A retains, not released
    with pytest.raises(PermissionError):
        provs["A"].release(stale)                                       # replayed by anyone: refused
    assert (owner.own, row.tree_id) in provs["A"].retained
