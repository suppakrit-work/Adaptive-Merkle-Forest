"""Shared set-up for the tests: one owner, three providers, three types (Fig. 1 in miniature)."""
import random

from amf.common.merkle import MerkleTree
from amf.owner import Owner
from amf.provider import Provider

TYPES = ("image", "video", "document")


def world(s=4, seed=1, owner=None):
    """image -> A, video -> B, document -> C."""
    owner = owner or Owner(s=s, rng=random.Random(seed))
    provs = {n: Provider(n) for n in "ABC"}
    for p in provs.values():
        owner.register_provider(p)
    for tag, (ftype, prov) in enumerate(zip(TYPES, "ABC"), start=1):
        owner.add_type(ftype, tag, prov)
    return owner, provs


def fill(owner, n=10, ftype=None, prefix="f"):
    files = {}
    for i in range(n):
        t = ftype or TYPES[i % 3]
        name, data = f"{prefix}{t}-{i}", (b"content of %d " % i) * (i % 7 + 1)
        owner.upload(name, data, t)
        files[name] = data
    return files


def tree_of(owner, provs, ftype):
    """(provider, row) of the single live tree of a type."""
    lo_tag = owner.type_table[ftype]
    rows = [r for r in owner.live_rows() if (r.lo >> 256) & 0xFF == lo_tag]
    (row,) = rows
    return provs[row.provider], row


def tamper_stored_leaf(prov, own, tree_id, i, value=b"\x00" * 32):
    """Rewrite one stored leaf value in the Merkle Tree Index, and the nodes above it."""
    t = prov.live[(own, tree_id)]
    t.leaves[i] = value
    t.tree = MerkleTree(t.leaves)