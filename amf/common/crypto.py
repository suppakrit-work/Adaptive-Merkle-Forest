"""Object Commitment -- Phase 2 of Section III-C.

    id   = tag || HMAC(K_id, name)      tag = owner's tag || type tag
    k    = HKDF(MK, id)
    CT   = IV || AES-256-GCM(k, IV, file)
    leaf = H(id || CT)                  H = SHA-256

Every owner is known to the others by a tag, the hash of its public key, which
prefixes every identifier of its objects, so the identifier spaces of
different owners are disjoint.  K_id is a key derived from MK for identifiers
alone; the IV is fresh for every encryption.  The identifier goes into the
leaf so that a reader who holds the identifier can tell that the leaf it is
shown belongs to that object.  The provider gets the pair (id, CT).
"""
import hashlib
import os

from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

OWN_LEN = 32         # owner's tag: SHA-256 of the raw Ed25519 public key
TYPE_LEN = 1         # one type-tag byte
TAG_LEN = OWN_LEN + TYPE_LEN
HMAC_LEN = 32        # HMAC-SHA256 output
ID_LEN = TAG_LEN + HMAC_LEN          # 65 bytes
IV_LEN = 12          # the standard GCM nonce length
KEY_LEN = 32         # AES-256
GCM_TAG_LEN = 16


def _hkdf(mk: bytes, info: bytes, length: int = KEY_LEN) -> bytes:
    """HKDF-SHA256 over the master key with a context string."""
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=None, info=info).derive(mk)


def owner_tag(pk) -> bytes:
    """The owner's tag: the hash of its public key (32 bytes)."""
    raw = pk.public_bytes(Encoding.Raw, PublicFormat.Raw)
    return hashlib.sha256(raw).digest()


def derive_id_key(mk: bytes) -> bytes:
    """K_id: the key used for identifiers alone."""
    return _hkdf(mk, b"amf/id-key")


def make_id(k_id: bytes, own: bytes, type_tag: int, name: str) -> bytes:
    """id = own || type tag || HMAC(K_id, name).  65 bytes."""
    if len(own) != OWN_LEN:
        raise ValueError("the owner's tag is 32 bytes")
    if not 0 <= type_tag <= 255:
        raise ValueError("a type tag must fit in one byte")
    h = hmac.HMAC(k_id, hashes.SHA256())
    h.update(name.encode("utf-8"))
    mac = h.finalize()
    return own + bytes([type_tag]) + mac


def owner_of(obj_id: bytes) -> bytes:
    """The owner's tag an identifier starts with."""
    if len(obj_id) != ID_LEN:
        raise ValueError("not a 65-byte identifier")
    return obj_id[:OWN_LEN]


def type_of(obj_id: bytes) -> int:
    if len(obj_id) != ID_LEN:
        raise ValueError("not a 65-byte identifier")
    return obj_id[OWN_LEN]


def derive_object_key(mk: bytes, obj_id: bytes) -> bytes:
    """k = HKDF(MK, id).  The key exists for this object only and is never written down."""
    return _hkdf(mk, b"amf/object-key" + obj_id)


def encrypt(mk: bytes, obj_id: bytes, data: bytes) -> bytes:
    """CT = IV || AES-256-GCM(k, IV, data), with a fresh IV every call."""
    k = derive_object_key(mk, obj_id)
    iv = os.urandom(IV_LEN)
    return iv + AESGCM(k).encrypt(iv, data, None)


def decrypt(mk: bytes, obj_id: bytes, ct: bytes) -> bytes:
    """Inverse of encrypt.  Raises cryptography.exceptions.InvalidTag on tampering."""
    k = derive_object_key(mk, obj_id)
    iv, body = ct[:IV_LEN], ct[IV_LEN:]
    return AESGCM(k).decrypt(iv, body, None)


def leaf_hash(obj_id: bytes, ct: bytes) -> bytes:
    """leaf = SHA-256(id || CT): binds the bytes the provider keeps to the identifier."""
    if len(obj_id) != ID_LEN:
        raise ValueError("not a 65-byte identifier")
    return hashlib.sha256(obj_id + ct).digest()
