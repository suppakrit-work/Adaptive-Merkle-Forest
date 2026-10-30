"""What goes over the network between the owner, readers, auditors and providers: JSON, never pickle.

JSON has no bytes, tuples or non-string keys, so those are tagged:

    bytes                {"$b": base64}
    tuple                {"$t": [...]}
    dict, other keys     {"$d": [[key, value], ...]}
    index row            {"$row": {field: value}}
    forest digest        {"$digest": {field: value}}
    signed order         {"$order": {field: value}}
    Ed25519 public key   {"$pk": base64 of the raw 32 bytes}

Integers of any size are plain JSON numbers.  Decoding knows these tags and
nothing else, and a tagged record must carry exactly the fields of its class,
so a peer cannot make the other side build an object it did not ask for.
Whatever arrives is still only data: rows, digests and orders are checked
against the owner's signature by whoever receives them, as before.
"""
import base64
import json
from dataclasses import fields

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from amf.common.index import ForestDigest, IndexRow
from amf.common.orders import Order

RECORDS = {"$row": IndexRow, "$digest": ForestDigest, "$order": Order}


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def to_wire(x):
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (bytes, bytearray)):
        return {"$b": _b64(bytes(x))}
    if isinstance(x, tuple):
        return {"$t": [to_wire(v) for v in x]}
    if isinstance(x, (list, range)):
        return [to_wire(v) for v in x]
    if isinstance(x, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in x):
            return {k: to_wire(v) for k, v in x.items()}
        return {"$d": [[to_wire(k), to_wire(v)] for k, v in x.items()]}
    for tag, cls in RECORDS.items():
        if type(x) is cls:
            return {tag: {f.name: to_wire(getattr(x, f.name)) for f in fields(cls)}}
    if isinstance(x, Ed25519PublicKey):
        return {"$pk": _b64(x.public_bytes(Encoding.Raw, PublicFormat.Raw))}
    raise TypeError(f"cannot send a {type(x).__name__}")


def from_wire(x):
    if isinstance(x, list):
        return [from_wire(v) for v in x]
    if not isinstance(x, dict):
        return x
    tags = [k for k in x if k.startswith("$")]
    if not tags:
        return {k: from_wire(v) for k, v in x.items()}
    if len(x) != 1:
        raise ValueError("a tagged value has exactly one key")
    (tag, v), = x.items()
    if tag == "$b":
        return base64.b64decode(v, validate=True)
    if tag == "$t":
        return tuple(from_wire(i) for i in v)
    if tag == "$d":
        return {from_wire(k): from_wire(val) for k, val in v}
    if tag == "$pk":
        return Ed25519PublicKey.from_public_bytes(base64.b64decode(v, validate=True))
    if tag in RECORDS:
        cls = RECORDS[tag]
        if not isinstance(v, dict) or set(v) != {f.name for f in fields(cls)}:
            raise ValueError(f"{tag} does not carry the fields of {cls.__name__}")
        return cls(**{k: from_wire(val) for k, val in v.items()})
    raise ValueError(f"unknown tag {tag}")


def dumps(x) -> bytes:
    return json.dumps(to_wire(x), separators=(",", ":")).encode("utf-8")


def loads(data: bytes):
    return from_wire(json.loads(data.decode("utf-8")))