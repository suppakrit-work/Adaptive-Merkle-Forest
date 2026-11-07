"""The whole protocol over HTTP: three provider services on localhost, the owner, a reader and an auditor."""
import random
import threading

import pytest

from amf.common import crypto
from amf.auditor import Auditor, Reader
from amf.owner.localize import Localizer
from amf.common.merkle import node_hash
from amf.common.orders import Order
from amf.owner import Owner, ReadFault
from amf.provider.client import RemoteError, RemoteProvider
from amf.provider.server import make_server
from tests.helpers import TYPES, fill


@pytest.fixture
def net(tmp_path):
    """Three providers A, B, C, each an HTTP service with its objects on disk; peers for the Migrate Pipe."""
    servers = {}
    for name in "ABC":
        srv, svc = make_server(name, "127.0.0.1", 0, str(tmp_path / name), peers={}, debug=True, sync=False)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        servers[name] = (srv, svc)
    urls = {n: f"http://127.0.0.1:{srv.server_address[1]}" for n, (srv, _) in servers.items()}
    for n, (_, svc) in servers.items():
        svc.peers.update({m: RemoteProvider(m, u) for m, u in urls.items() if m != n})
    provs = {n: RemoteProvider(n, u) for n, u in urls.items()}
    yield provs, {n: svc for n, (_, svc) in servers.items()}
    for srv, svc in servers.values():
        srv.shutdown()
        srv.server_close()
        for peer in svc.peers.values():
            peer.net.close()
    for p in provs.values():
        p.net.close()


def remote_world(provs, owner=None):
    owner = owner or Owner(s=4, rng=random.Random(3))
    for p in provs.values():
        owner.register_provider(p)
    for tag, (ftype, prov) in enumerate(zip(TYPES, "ABC"), start=1):
        owner.add_type(ftype, tag, prov)
    return owner


def test_every_service_answers_health_and_stats(net):
    provs, _ = net
    assert [p.health()["name"] for p in provs.values()] == ["A", "B", "C"]
    assert provs["A"].stats()["objects"] == 0


def test_uploads_and_reads_go_over_http_and_objects_land_on_disk(net):
    provs, svcs = net
    owner = remote_world(provs)
    files = fill(owner, 12)
    for name, data in files.items():
        assert owner.read(name) == data
    assert [len(svcs[n].provider.objects) for n in "ABC"] == [4, 4, 4]
    assert all(svcs[n].provider.dht.digest(owner.own) == owner.digest for n in "ABC")


def test_split_and_merge_are_checked_with_one_hash_over_http(net):
    provs, _ = net
    owner = remote_world(provs)
    files = fill(owner, 20, ftype="video")
    (row,) = [r for r in owner.live_rows() if r.provider == "B"]
    owner.hashes = 0
    a, b, _ = owner.split(row.tree_id)
    assert owner.hashes == 1 and node_hash(a.root, b.root) == row.root
    ab = owner.merge(a.tree_id, b.tree_id)
    assert owner.hashes == 2 and ab.root == row.root
    assert all(owner.read(n) == d for n, d in files.items())


def test_a_tree_moves_through_the_pipe_and_the_source_drops_it_on_release(net):
    provs, svcs = net
    owner = remote_world(provs)
    files = fill(owner, 9, ftype="image")
    (row,) = [r for r in owner.live_rows() if r.provider == "A"]
    moved = owner.migrate(row.tree_id, "C")                       # C pulls the tree from A itself
    assert moved.provider == "C" and len(svcs["C"].provider.objects) == 9
    assert len(svcs["A"].provider.retained) == 1 and len(svcs["A"].provider.objects) == 9
    assert all(owner.read(n) == d for n, d in files.items())
    assert owner.release(row.tree_id, "A") == 9 and len(svcs["A"].provider.objects) == 0


def test_a_damaged_object_is_named_over_http(net):
    provs, _ = net
    owner = remote_world(provs)
    fill(owner, 30, ftype="document")
    (row,) = [r for r in owner.live_rows() if r.provider == "C"]
    victim = owner.catalog.id_at(row.lo, 11)
    provs["C"].corrupt(victim, 7)                                  # the service runs with --debug
    with pytest.raises(ReadFault) as e:
        owner.read(owner.catalog.name_of(victim))
    assert (e.value.provider, e.value.position, e.value.kind) == ("C", 11, "ciphertext")
    loc = Localizer(owner)
    rep = loc.localize(row.tree_id)
    assert rep.verdict == "objects" and rep.names == [owner.catalog.name_of(victim)] and rep.rounds == 5
    assert loc.localize(row.tree_id, one_exchange=True).positions == [11]


def test_an_auditor_and_a_data_user_need_only_public_keys_and_d(net):
    provs, _ = net
    owner = remote_world(provs)
    files = fill(owner, 15)
    plist = list(provs.values())
    audit = Auditor([owner.pk], s=3, rng=random.Random(1)).audit_owner(owner.digest, plist)
    assert audit.ok and audit.digest_ok and len(audit.trees) == 3
    obj_id, d = owner.reader_view("fimage-3", "image")
    ct = Reader([owner.pk]).read(obj_id, d, plist)
    assert crypto.decrypt(owner.mk, obj_id, ct) == files["fimage-3"]


def test_the_owner_is_rebuilt_from_remote_providers(net):
    provs, _ = net
    owner = remote_world(provs)
    files = fill(owner, 12)
    owner.delete("fvideo-1")
    del files["fvideo-1"]
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    assert back.digest == owner.digest and back.type_table == owner.type_table
    assert all(back.read(n) == d for n, d in files.items())


def test_errors_keep_their_meaning_across_the_network(net):
    provs, svcs = net
    owner = remote_world(provs)
    fill(owner, 6, ftype="image")
    (row,) = [r for r in owner.live_rows() if r.provider == "A"]
    unsigned = Order("split", owner.own, (row.tree_id,), (b"a" * 16, b"b" * 16), owner.median_boundary(row), "A",
                     version=row.version)
    with pytest.raises(PermissionError):
        provs["A"].decompose(unsigned)                             # refused at the provider, as locally
    with pytest.raises(RemoteError):
        provs["A"].call("provider", "__init__", "X")                # not a protocol method
    svcs["A"].debug = False
    with pytest.raises(RemoteError):
        provs["A"].corrupt(owner.catalog.id_at(row.lo, 0))          # fault injection is off


def test_every_call_counts_its_bytes(net):
    provs, _ = net
    owner = remote_world(provs)
    before = provs["A"].net.received
    fill(owner, 3, ftype="image")
    owner.read("fimage-0")
    assert provs["A"].net.calls > 0 and provs["A"].net.received > before


# ---- sending to every DHT node at once (III-C, Parallelism) ----------------------------------

def test_an_owner_that_sends_to_every_node_at_once_keeps_the_nodes_in_step(net):
    provs, svcs = net
    owner = remote_world(provs, Owner(s=4, rng=random.Random(3), parallel=True))
    files = fill(owner, 30)
    a, b, _ = owner.split(next(r for r in owner.live_rows() if r.provider == "A").tree_id)
    owner.migrate(b.tree_id, "C")
    for n in "ABC":
        node = svcs[n].provider.dht
        assert node.digest(owner.own) == owner.digest                  # every node has the latest D
        assert {r.tree_id: r for r in node.live_rows(owner.own)} == {r.tree_id: r for r in owner.live_rows()}
    assert all(owner.read(name) == data for name, data in files.items())
    back = Owner.recover(owner.mk, owner.sk, list(provs.values()))
    assert back.digest == owner.digest and back.read("fimage-0") == files["fimage-0"]


def test_at_once_or_one_at_a_time_the_messages_are_the_same(net):
    provs, _ = net
    counts = []
    for parallel in (False, True):
        owner = remote_world(provs, Owner(s=4, rng=random.Random(3), parallel=parallel))
        before = sum(p.net.calls for p in provs.values())
        owner.upload("one.jpg", b"same bytes", "image")
        counts.append(sum(p.net.calls for p in provs.values()) - before)
    assert counts[0] == counts[1] == 8                                 # only the waiting changes



def test_two_migrations_in_opposite_directions_do_not_wait_for_each_other(net):
    provs, _ = net
    w = remote_world(provs)
    v = remote_world(provs, Owner(s=4, rng=random.Random(4)))
    fill(w, 9, ftype="image")                                          # W's images at A
    fill(v, 9, ftype="video")                                          # V's videos at B
    (wa,) = [r for r in w.live_rows() if r.provider == "A"]
    (vb,) = [r for r in v.live_rows() if r.provider == "B"]
    done = []
    jobs = [threading.Thread(target=lambda: done.append(w.migrate(wa.tree_id, "B")), daemon=True),
            threading.Thread(target=lambda: done.append(v.migrate(vb.tree_id, "A")), daemon=True)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join(timeout=20)
    assert not any(j.is_alive() for j in jobs) and sorted(r.provider for r in done) == ["A", "B"]


def test_a_batch_write_goes_over_http_as_one_call_per_tree(net):
    provs, svcs = net
    owner = remote_world(provs)
    files = [(f"b{i}", b"batch %d" % i, TYPES[i % 3]) for i in range(30)]
    calls0 = provs["A"].net.calls
    rows = owner.upload_many(files)
    assert len(rows) == 3                                            # one batch per type, one tree each
    assert provs["A"].net.calls - calls0 < 30                        # far fewer calls than objects
    for name, data, _t in files:
        assert owner.read(name) == data
    assert svcs["A"].provider.stats()["objects"] == 10 and svcs["A"].provider.stats()["shadow"] == 0
