"""Signed orders -- Phase 4, step 1, and the Migrate Pipe (III-A 3).

"The owner sends a signed order to one provider.  For a decomposition it names
the tree, the boundary p and two new tree identifiers; for a composition the
two trees and one new identifier; for a migration the tree, its source, its
destination and the row version it applies to, and it goes to the
destination."  A migration source keeps its copy until a second signed order
from the owner (kind "release").  A provider acts on an order only when the
signature verifies under the public key of the owner the order names.
"""
from dataclasses import dataclass, replace

from cryptography.exceptions import InvalidSignature

from amf.common.crypto import OWN_LEN, owner_tag
from amf.common.index import BOUND_LEN, TREE_ID_LEN

DOMAIN = b"amf/order/v1"
KINDS = ("split", "merge", "migrate", "release")


@dataclass(frozen=True)
class Order:
    kind: str                    # split | merge | migrate | release
    own: bytes                   # the owner whose trees these are
    trees: tuple                 # tree ids the order acts on
    new_ids: tuple = ()          # fresh tree ids: two for a split, one for a merge
    boundary: int = 0            # p, for a split
    source: str = ""             # the provider holding the tree(s)
    destination: str = ""        # for a migration
    version: int = 0             # the row version the order applies to
    signature: bytes = b""

    def payload(self) -> bytes:
        if self.kind not in KINDS:
            raise ValueError("unknown order kind")
        src, dst = self.source.encode(), self.destination.encode()
        return (DOMAIN + bytes([KINDS.index(self.kind)]) + self.own
                + bytes([len(self.trees)]) + b"".join(self.trees)
                + bytes([len(self.new_ids)]) + b"".join(self.new_ids)
                + self.boundary.to_bytes(BOUND_LEN, "big")
                + bytes([len(src)]) + src + bytes([len(dst)]) + dst
                + self.version.to_bytes(8, "big"))


def sign_order(sk, order: Order) -> Order:
    return replace(order, signature=sk.sign(order.payload()))


def verify_order(pk, order: Order) -> bool:
    try:
        if pk is None or len(order.own) != OWN_LEN or order.own != owner_tag(pk) or len(order.signature) != 64:
            return False
        if any(len(t) != TREE_ID_LEN for t in order.trees + order.new_ids):
            return False
        pk.verify(order.signature, order.payload())
        return True
    except (InvalidSignature, OverflowError, ValueError, TypeError, AttributeError):
        return False
