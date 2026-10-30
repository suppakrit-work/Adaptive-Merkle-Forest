"""Tests for amf/orders.py -- the owner's signed orders (Phase 4 step 1, Migrate Pipe)."""
from dataclasses import replace

from amf.common import crypto
from amf.common.index import generate_signing_key, new_tree_id
from amf.common.orders import Order, sign_order, verify_order

SK = generate_signing_key()
PK = SK.public_key()
OWN = crypto.owner_tag(PK)
OTHER = generate_signing_key()


def test_each_kind_of_order_verifies_under_the_owners_key():
    t, a, b = new_tree_id(), new_tree_id(), new_tree_id()
    orders = [Order("split", OWN, (t,), (a, b), boundary=12345, source="A", version=3),
              Order("merge", OWN, (a, b), (t,), source="A", version=1),
              Order("migrate", OWN, (t,), source="A", destination="C", version=4),
              Order("release", OWN, (t,), source="A", version=5)]
    for o in orders:
        assert verify_order(PK, sign_order(SK, o))


def test_every_field_is_covered_by_the_signature():
    t = new_tree_id()
    o = sign_order(SK, Order("migrate", OWN, (t,), source="A", destination="C", version=4))
    for bad in (replace(o, destination="B"), replace(o, source="B"), replace(o, version=5),
                replace(o, trees=(new_tree_id(),)), replace(o, kind="release"), replace(o, boundary=1)):
        assert not verify_order(PK, bad)


def test_only_the_owner_can_order():
    t = new_tree_id()
    assert not verify_order(PK, sign_order(OTHER, Order("release", OWN, (t,), source="A")))
    assert not verify_order(PK, Order("release", OWN, (t,), source="A"))       # unsigned
    assert not verify_order(None, sign_order(SK, Order("release", OWN, (t,), source="A")))