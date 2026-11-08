"""Tests for amf/wire.py -- JSON on the wire, nothing that could run code on the other side."""
import json

import pytest

from amf.common import wire
from amf.common.index import ForestDigest, IndexRow, generate_signing_key, new_tree_id, sign_row, tag_range, verify_row
from amf.common.crypto import owner_tag
from amf.common.orders import Order, sign_order, verify_order

SK = generate_signing_key()
PK = SK.public_key()
OWN = owner_tag(PK)


def a_row():
    lo, hi = tag_range(OWN, 1)
    return sign_row(SK, IndexRow(OWN, new_tree_id(), lo, hi, "A", b"\x05" * 32, 12, 3))


def test_plain_values_and_bytes_round_trip():
    for x in (None, True, 0, -7, 2 ** 600, "text", b"", b"\x00\xff" * 40, [1, b"x", [None]]):
        assert wire.loads(wire.dumps(x)) == x


def test_a_float_is_refused_on_the_wire():
    """A count of 3.0 is not a count: nothing in the protocol is a float, so one is a malformed answer."""
    with pytest.raises(ValueError):
        wire.loads(b'{"result": [1, 2.0]}')
    with pytest.raises(ValueError):
        wire.loads(b'3.5')


def test_tuples_stay_tuples_and_dicts_may_have_any_keys():
    x = {(b"own", b"tid"): (3, 1), "plain": {"k": (1, 2)}, "$looks-like-a-tag": 1, 5: b"v"}
    back = wire.loads(wire.dumps(x))
    assert back == x and isinstance(back[(b"own", b"tid")], tuple)


def test_a_signed_row_still_verifies_after_the_trip():
    row = a_row()
    back = wire.loads(wire.dumps(row))
    assert back == row and verify_row(PK, back)


def test_digests_orders_and_public_keys_round_trip():
    d = ForestDigest(OWN, b"\x01" * 32, 4, 9, b"\x02" * 64)
    order = sign_order(SK, Order("split", OWN, (new_tree_id(),), (new_tree_id(), new_tree_id()), 2 ** 500, "A",
                                 version=3))
    d2, o2, pk2 = wire.loads(wire.dumps([d, order, PK]))
    assert d2 == d and o2 == order and verify_order(pk2, o2) and owner_tag(pk2) == OWN


def test_ranges_go_as_lists():
    assert wire.loads(wire.dumps(range(4))) == [0, 1, 2, 3]


def test_unknown_tags_and_incomplete_records_are_refused():
    for bad in ({"$pickle": "x"}, {"$b": "x", "extra": 1}, {"$row": {"own": {"$b": ""}}},
                {"$b": "not base64!"}):
        with pytest.raises(ValueError):
            wire.from_wire(bad)


def test_what_cannot_be_sent_is_refused():
    with pytest.raises(TypeError):
        wire.dumps(object())
    with pytest.raises(TypeError):
        wire.dumps({1, 2})


def test_the_bytes_on_the_wire_are_plain_json():
    data = wire.dumps({"args": [a_row(), (1, 2)]})
    assert isinstance(json.loads(data), dict)
