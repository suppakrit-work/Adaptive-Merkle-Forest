"""A reshaping without a check: what a verifier of a traffic-aware tree (Mizrahi et al., ToN 2024)
or of HMT (Shangguan et al., 2026) must do to be sure of a new root.

A. Mizrahi, N. Koren, O. Rottenstreich and Y. Cassuto, "Traffic-Aware Merkle
Trees for Shortening Blockchain Transaction Proofs", IEEE/ACM Trans. Netw.
32(6), 2024, and Z. Shangguan, A. Yaish and D. Malkhi, "Authenticated Data
Structures for Dynamic Workloads", Cryptology ePrint 2026/1235, reshape a tree
as access patterns change: Huffman-shaped layouts, rebuilt or swapped at
milestones from observed frequencies.  The party that reshapes publishes the
new root.  Nothing in these schemes lets a verifier that holds only the old
root check the new one cheaply; the restructuring is trusted, or the verifier
downloads the elements and rebuilds the root itself.

This module measures that second option for the restructuring that AMF
checks with one hash, a split at the median: the time to compute the two new
roots from the tree's objects and the bytes the verifier must download for
it.  It does not build Huffman layouts (the HMT baseline, shangguan2026.py,
does); it is the cost of verifying a reshaping by rebuilding, whatever the
new shape is.
"""
import time

from amf.common import crypto
from amf.common.merkle import root_of


def rebuild_at_verifier(ids: list, objects: dict, split_at: int) -> tuple[bytes, bytes, float, int]:
    """The verifier downloads every object of the tree (ids in tree order, objects: id -> CT),
    hashes the leaves and builds the two roots of a split at position split_at.
    Returns (root of the first part, root of the second part, milliseconds, bytes downloaded)."""
    cts = [objects[i] for i in ids]
    t0 = time.perf_counter()
    leaves = [crypto.leaf_hash(i, ct) for i, ct in zip(ids, cts)]
    r_a, r_b = root_of(leaves[:split_at]), root_of(leaves[split_at:])
    ms = 1000 * (time.perf_counter() - t0)
    return r_a, r_b, ms, sum(len(i) + len(ct) for i, ct in zip(ids, cts))
