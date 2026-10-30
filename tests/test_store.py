"""Tests for amf/store.py -- the provider's objects on disk."""
import os

import pytest

from amf.common.crypto import ID_LEN
from amf.provider import Provider
from amf.provider.store import FileStore
from tests.helpers import fill, world


def oid(k: int) -> bytes:
    return k.to_bytes(ID_LEN, "big")


def test_store_acts_like_a_dict(tmp_path):
    s = FileStore(str(tmp_path))
    s[oid(1)] = b"one"
    s.update({oid(2): b"two", oid(3): b"three"})
    assert s[oid(2)] == b"two" and oid(3) in s and oid(9) not in s and len(s) == 3
    assert sorted(s) == [oid(1), oid(2), oid(3)]
    assert s.pop(oid(1)) == b"one" and s.pop(oid(1), None) is None
    del s[oid(2)]
    with pytest.raises(KeyError):
        s[oid(2)]
    with pytest.raises(KeyError):
        del s[oid(2)]


def test_a_rewrite_replaces_the_whole_file(tmp_path):
    s = FileStore(str(tmp_path))
    s[oid(1)] = b"a much longer first version"
    s[oid(1)] = b"short"
    assert s[oid(1)] == b"short"
    assert not [n for n in os.listdir(tmp_path) if n.endswith(".tmp")]


def test_a_leftover_temporary_file_is_not_an_object(tmp_path):
    s = FileStore(str(tmp_path))
    s[oid(1)] = b"kept"
    (tmp_path / (oid(2).hex() + ".tmp")).write_bytes(b"half written")
    assert list(s) == [oid(1)] and len(s) == 1


def test_only_identifiers_are_keys(tmp_path):
    s = FileStore(str(tmp_path))
    with pytest.raises(KeyError):
        s[b"../../etc/passwd"] = b"x"
    assert "not bytes" not in s


def test_a_provider_keeps_its_objects_on_disk(tmp_path):
    owner, provs = world()
    disk = Provider("D", objects=FileStore(str(tmp_path), sync=False))
    owner.register_provider(disk)
    owner.add_type("archive", 7, "D")
    files = fill(owner, 6, ftype="archive")
    assert len(os.listdir(tmp_path)) == 6
    for name, data in files.items():
        assert owner.read(name) == data
    tid = owner.row_for(owner.catalog.get("farchive-0")[0]).tree_id
    owner.migrate(tid, "A")                                   # D keeps its copy until released
    assert len(os.listdir(tmp_path)) == 6
    assert owner.release(tid, "D") == 6 and os.listdir(tmp_path) == []