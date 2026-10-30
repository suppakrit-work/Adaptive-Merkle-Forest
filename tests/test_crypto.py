"""Tests for amf/crypto.py -- Phase 2, Object Commitment.

Run with:  python -m pytest -q
Every test here states one sentence of the paper and checks it.
"""
import hashlib
import hmac
import os

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from amf.common import crypto

MK = bytes(range(32))          # a fixed master key so tests are repeatable
MK2 = bytes(range(1, 33))      # a different master key
OWN = hashlib.sha256(b"owner W").digest()
OWN2 = hashlib.sha256(b"owner V").digest()


def ident(name, type_tag=0x01, mk=MK, own=OWN):
    return crypto.make_id(crypto.derive_id_key(mk), own, type_tag, name)


# ---- the owner's tag: the hash of its public key ------------------------------

def test_owner_tag_is_the_hash_of_the_public_key():
    pk = Ed25519PrivateKey.generate().public_key()
    raw = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
    assert crypto.owner_tag(pk) == hashlib.sha256(raw).digest()
    assert len(crypto.owner_tag(pk)) == crypto.OWN_LEN == 32
    assert crypto.owner_tag(pk) != crypto.owner_tag(Ed25519PrivateKey.generate().public_key())


# ---- id = own || type tag || HMAC(K_id, name) -----------------------------------

def test_id_is_owner_tag_type_tag_and_hmac():
    k_id = crypto.derive_id_key(MK)
    obj_id = crypto.make_id(k_id, OWN, 0x01, "holiday.jpg")
    assert len(obj_id) == crypto.ID_LEN == 65
    assert obj_id[:32] == OWN and obj_id[32] == 0x01
    assert obj_id[33:] == hmac.new(k_id, b"holiday.jpg", hashlib.sha256).digest()
    assert crypto.owner_of(obj_id) == OWN and crypto.type_of(obj_id) == 0x01


def test_identifier_spaces_of_different_owners_are_disjoint():
    # "which prefixes every identifier of its objects, so the identifier spaces
    #  of different owners are disjoint"
    a, b = ident("x", own=OWN), ident("x", own=OWN2)
    assert a[:32] != b[:32]


def test_id_is_deterministic_and_name_sensitive():
    a, b, c = ident("holiday.jpg"), ident("holiday.jpg"), ident("holiday.jpeg")
    d = ident("holiday.jpg", type_tag=0x02)
    assert a == b and a != c
    assert a[33:] == d[33:] and a[32] != d[32]            # only the type tag differs


def test_id_depends_on_master_key():
    assert ident("holiday.jpg", mk=MK) != ident("holiday.jpg", mk=MK2)


def test_id_key_is_not_the_master_key():
    # "a key derived from MK for identifiers alone"
    assert crypto.derive_id_key(MK) != MK and len(crypto.derive_id_key(MK)) == 32


def test_tags_must_have_the_right_size():
    with pytest.raises(ValueError):
        crypto.make_id(crypto.derive_id_key(MK), OWN, 256, "x")
    with pytest.raises(ValueError):
        crypto.make_id(crypto.derive_id_key(MK), OWN[:8], 1, "x")


# ---- k = HKDF(MK, id) ------------------------------------------------------

def test_object_key_per_object():
    id_a, id_b = ident("a"), ident("b")
    assert len(crypto.derive_object_key(MK, id_a)) == 32
    assert crypto.derive_object_key(MK, id_a) == crypto.derive_object_key(MK, id_a)
    assert crypto.derive_object_key(MK, id_a) != crypto.derive_object_key(MK, id_b)
    assert crypto.derive_object_key(MK, id_a) != crypto.derive_object_key(MK2, id_a)
    assert crypto.derive_object_key(MK, id_a) != crypto.derive_id_key(MK)


# ---- CT = IV || AES-256-GCM(k, IV, file) ----------------------------------

def test_ciphertext_layout_and_roundtrip():
    obj_id, data = ident("holiday.jpg"), os.urandom(1000)
    ct = crypto.encrypt(MK, obj_id, data)
    assert len(ct) == crypto.IV_LEN + len(data) + crypto.GCM_TAG_LEN
    assert crypto.decrypt(MK, obj_id, ct) == data


def test_iv_is_fresh_per_encryption():
    # "the IV is fresh for every encryption, since an object rewritten under
    #  the same name reuses its key"
    obj_id = ident("holiday.jpg")
    ct1, ct2 = crypto.encrypt(MK, obj_id, b"same bytes"), crypto.encrypt(MK, obj_id, b"same bytes")
    assert ct1[:crypto.IV_LEN] != ct2[:crypto.IV_LEN] and ct1 != ct2


def test_tampering_is_detected():
    obj_id = ident("holiday.jpg")
    ct = bytearray(crypto.encrypt(MK, obj_id, b"hello provider"))
    ct[crypto.IV_LEN + 3] ^= 0x01
    with pytest.raises(InvalidTag):
        crypto.decrypt(MK, obj_id, bytes(ct))


def test_wrong_object_wrong_key():
    id_a, id_b = ident("a"), ident("b")
    ct = crypto.encrypt(MK, id_a, b"secret")
    with pytest.raises(InvalidTag):
        crypto.decrypt(MK, id_b, ct)


def test_plaintext_never_appears_in_ciphertext():
    ct = crypto.encrypt(MK, ident("holiday.jpg"), b"THIS IS THE SECRET CONTENT OF THE FILE " * 4)
    assert b"SECRET" not in ct


# ---- leaf = H(id || CT) ------------------------------------------------------

def test_leaf_is_sha256_of_id_and_ciphertext():
    obj_id = ident("holiday.jpg")
    ct = crypto.encrypt(MK, obj_id, b"payload")
    assert crypto.leaf_hash(obj_id, ct) == hashlib.sha256(obj_id + ct).digest()
    changed = bytearray(ct)
    changed[-1] ^= 0xFF
    assert crypto.leaf_hash(obj_id, bytes(changed)) != crypto.leaf_hash(obj_id, ct)


def test_the_leaf_belongs_to_one_object():
    # "a reader who holds the identifier can tell that the leaf it is shown
    #  belongs to that object and not to another one the provider would rather show"
    id_a, id_b = ident("a"), ident("b")
    ct = crypto.encrypt(MK, id_a, b"same bytes")
    assert crypto.leaf_hash(id_a, ct) != crypto.leaf_hash(id_b, ct)