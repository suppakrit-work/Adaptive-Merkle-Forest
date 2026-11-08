"""Tests for amf/policy.py -- the split and merge rules of Phase 4."""
import time

import pytest

from amf.owner import Owner
from amf.owner.policy import PolicyManager, hot
from tests.helpers import fill, tree_of, world


def test_hot_needs_enough_accesses_and_one_half_much_busier():
    assert hot(45, 5, gamma=4, mu=50)                  # 50 accesses, 45 > 4 * 5
    assert not hot(40, 9, gamma=4, mu=50)              # not enough accesses yet
    assert not hot(40, 10, gamma=4, mu=50)             # 40 is not more than 4 * 10
    assert hot(50, 0, gamma=4, mu=50)


def test_n_low_must_stay_well_under_half_of_n_up():
    owner, _ = world()
    with pytest.raises(ValueError):
        PolicyManager(owner, upper=100, lower=30)


def test_the_size_test_runs_before_a_write_so_no_tree_passes_n_up():
    owner, provs = world()
    PolicyManager(owner, upper=20, lower=5)
    fill(owner, 60, ftype="image")
    rows = [r for r in owner.live_rows() if r.provider == "A"]
    assert len(rows) > 1 and all(r.count <= 20 for r in rows)
    assert sum(r.count for r in rows) == 60


def test_split_when_the_halves_are_hot_and_the_tree_has_at_least_2_n_low():
    owner, provs = world()
    pm = PolicyManager(owner, upper=100, lower=10, gamma=4, mu=20)
    fill(owner, 30, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.write_counts.clear()
    names = [owner.catalog.name_of(i) for i in owner.catalog.ids_in(row.lo, row.hi)]
    for _ in range(25):
        owner.read(names[0])                           # the left half only
    (d,) = pm.tick()
    assert d.kind == "split" and d.trees == (row.tree_id,)
    assert owner.index[row.tree_id].retired


def test_no_hot_split_below_2_n_low():
    owner, provs = world()
    pm = PolicyManager(owner, upper=100, lower=20, gamma=4, mu=20)
    fill(owner, 30, ftype="image")                     # 30 < 2 * 20
    prov, row = tree_of(owner, provs, "image")
    owner.write_counts.clear()
    for _ in range(40):
        owner.read(owner.catalog.name_of(owner.catalog.id_at(row.lo, 0)))
    assert pm.tick() == []


def test_writes_count_as_accesses_too():
    owner, provs = world()
    pm = PolicyManager(owner, upper=200, lower=10, gamma=4, mu=20)
    fill(owner, 40, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.write_counts.clear()
    first = owner.catalog.name_of(owner.catalog.id_at(row.lo, 0))
    for k in range(25):
        owner.upload(first, b"rewrite %d" % k, "image")            # writes on the left half
    row = tree_of(owner, provs, "image")[1]
    assert pm.accesses(row.tree_id) == (25, 0)
    assert [d.kind for d in pm.tick()] == ["split"]


def test_merge_when_the_sum_falls_under_n_low_and_the_pair_is_not_hot():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=20, gamma=4, mu=10)
    fill(owner, 12, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    a, b, _ = owner.split(row.tree_id)
    owner.write_counts.clear()
    (d,) = pm.tick()
    assert d.kind == "merge" and set(d.trees) == {a.tree_id, b.tree_id}


def test_the_access_condition_keeps_two_trees_used_unevenly_apart():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=20, gamma=4, mu=10)
    fill(owner, 12, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    a, b, _ = owner.split(row.tree_id)
    owner.write_counts.clear()
    name = owner.catalog.name_of(owner.catalog.id_at(a.lo, 0))
    for _ in range(15):
        owner.read(name)                                           # only tree a is used
    assert pm.tick() == []


def test_merge_needs_the_same_provider():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=20, gamma=4, mu=10)
    fill(owner, 12, ftype="video")
    prov, row = tree_of(owner, provs, "video")
    a, b, _ = owner.split(row.tree_id)
    owner.migrate(b.tree_id, "C")
    owner.write_counts.clear()
    # a and b are no longer at one provider; b may merge with C's adjacent (document) tree instead,
    # since the type tag decides only where an object starts
    assert all(set(d.trees) != {a.tree_id, b.tree_id} for d in pm.tick())


def test_a_tree_is_not_touched_twice_within_delta_min_rounds():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=10, gamma=4, mu=20, interval=3)
    fill(owner, 40, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.write_counts.clear()
    for _ in range(30):
        owner.read(owner.catalog.name_of(owner.catalog.id_at(row.lo, 0)))
    (d,) = pm.tick()
    halves = [r for r in owner.live_rows() if r.provider == "A"]
    for _ in range(30):
        owner.read(owner.catalog.name_of(owner.catalog.id_at(halves[0].lo, 0)))
    assert pm.tick() == [] and pm.tick() == []                     # rounds 2 and 3: too soon
    assert [x.kind for x in pm.tick()] == ["split"]                # round 4


def test_the_policy_never_migrates_on_its_own():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=10, gamma=4, mu=5)
    fill(owner, 30)
    for _ in range(100):
        owner.read("fimage-0")                                     # A carries almost every read
    assert all(d.kind != "migrate" for d in pm.tick())


# ---- what the providers report, and a round in parallel -------------------------------------

def test_each_owner_gets_only_its_own_read_counts():
    w, provs = world()
    v = Owner()
    for p in provs.values():
        v.register_provider(p)
    v.add_type("image", 1, "A")
    for k in range(5):
        v.upload(f"v{k}", b"v's", "image")
    fill(w, 6, ftype="image")
    pw, pv = PolicyManager(w, upper=100, lower=10), PolicyManager(v, upper=100, lower=10)
    for _ in range(7):
        v.read("v0")
    pw.tick()                                                    # W's round must not use up V's counts
    pv.collect_reads()
    (vrow,) = v.live_rows()
    assert sum(pv.reads[vrow.tree_id]) == 7 and vrow.tree_id not in pw.reads


def test_a_provider_is_believed_only_about_its_own_trees_and_real_counts():
    owner, provs = world()
    pm = PolicyManager(owner, upper=100, lower=10, gamma=4, mu=20)
    fill(owner, 30, ftype="image")
    prov, row = tree_of(owner, provs, "image")
    owner.write_counts.clear()
    provs["B"].report_reads = lambda own: {row.key: (1000, 0)}       # B does not hold the image tree
    prov.report_reads = lambda own: {row.key: (-25, 0)}
    assert pm.tick() == [] and pm.accesses(row.tree_id) == (0, 0)


def test_a_round_sends_its_orders_out_together():
    owner, provs = world(owner=Owner(s=4, parallel=True))
    files = fill(owner, 75)                                      # 25 per type, one tree each
    pm = PolicyManager(owner, upper=20, lower=5)
    for p in provs.values():
        p.decompose = (lambda real: lambda order: (time.sleep(0.3), real(order))[1])(p.decompose)
    t0 = time.perf_counter()
    done = pm.tick()
    assert [d.kind for d in done] == ["split"] * 3 and time.perf_counter() - t0 < 0.75
    assert all(owner.read(n) == d for n, d in files.items())
    assert all(p.dht.digest(owner.own) == owner.digest for p in provs.values())



def test_the_policy_manager_orders_a_migration_after_a_failed_check():
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=10)
    files = fill(owner, 30)
    (moved,) = pm.failed_check("A")
    assert moved.provider != "A" and not pm.touchable(moved.tree_id)      # left alone for Delta_min rounds
    assert all(owner.read(n) == d for n, d in files.items())


def test_work_done_inside_a_parallel_round_is_not_booked_as_waiting():
    """C1 of the second review: with providers in-process the owner never waits, so the wait of a
    round with three decisions carried out in a pool must stay near zero."""
    from amf.common import waits
    owner, provs = world()
    owner.parallel = True
    pm = PolicyManager(owner, upper=1000, lower=250, mu=10)
    fill(owner, 60)
    for p in provs.values():
        p.report_reads = (lambda own, p=p: {k: (10 ** 6, 0) for k in p.live if k[0] == own})
    for r in owner.live_rows():                                      # make every tree hot and big enough
        pm.reads[r.tree_id] = [10 ** 6, 0]
    pm.lower = 5
    pm.upper = 1000
    w0 = waits.waited
    done = pm.tick()
    assert len(done) == 3                                             # three splits, carried out together
    assert waits.waited - w0 < 0.002                                  # seconds: the hashing was work, not waiting


def test_an_auditors_report_is_rechecked_before_the_owner_acts():
    """IV-F: the auditor is not trusted; the owner reads what the report names and keeps only what
    it sees itself."""
    from amf.auditor.auditor import AuditReport
    owner, provs = world()
    pm = PolicyManager(owner, upper=1000, lower=250)
    fill(owner, 12)
    prov, row = tree_of(owner, provs, "video")
    alarm = AuditReport("B", row.tree_id, bad_ciphertext=[1], missing=[2], bad_path=["x", 99])
    assert pm.on_audit(alarm) == []                                  # a false alarm: nothing confirmed
    assert owner.index[row.tree_id].provider == "B"
    prov.corrupt(prov.live[row.key].ids[2], 1)
    assert pm.on_audit(alarm) == [2]                                 # only the position the owner sees fail
    assert pm.on_audit(AuditReport("C", row.tree_id, bad_ciphertext=[2])) == []     # the wrong provider named
