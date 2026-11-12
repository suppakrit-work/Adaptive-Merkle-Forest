"""One Merkle hash tree per owner, as in Wang et al. (TPDS 2011): the baseline for a forest.

Q. Wang, C. Wang, K. Ren, W. Lou and J. Li, "Enabling Public Auditability and
Data Dynamics for Storage Security in Cloud Computing", IEEE Trans. Parallel
Distrib. Syst. 22(5), 2011, keep one Merkle hash tree (MHT) over all the
blocks of a file; the verifier holds the root.  A block is proved by its
authentication path.  An update is checked in two steps on the same path:
the old block and its path must lead to the root the verifier holds, then the
new root is computed from the new block along that path -- 2 log2 N hashes at
the verifier, log2 N at the server, whatever N is.  The scheme has no fault
localization: a failed proof says only that the tree is bad.  fetch-all, a
verifier that fetches every leaf hash to find the block, is our construction
of the cheapest thing the scheme allows, not something the paper proposes.

Taken from the paper: the single tree, the path proof and the path-update
check.  Left out: the BLS tags on the blocks and the third-party auditor's
public audits, which do not change the tree, its proofs or its updates; the
client's pairing check of the signature on the old root (our verifier holds
the root, the paper's client does not), its signature on the new root, and
the challenge of c random blocks the paper's client runs after every update
(Table 2), a possession check that AMF makes in audits, not on writes -- all
of which make the paper's client dearer than counted here.  The tree is AMF's Merkle tree
(amf.common.merkle), so both schemes hash the same bytes with the same
function; they differ in the number of trees and in how a change is checked
(AMF: Algorithm 3 against the owner's catalog; here: along the path).
"""
from amf.common.merkle import depth_of, node_hash, sides_of

HASH_LEN = 32


def depth(n: int) -> int:
    """The path length of every block in a tree of n blocks (ceil(log2 n))."""
    return depth_of(n)


def mean_path(n: int) -> float:
    """The mean path length over the n positions: every leaf of the canonical shape sits at depth
    d = ceil(log2 n) or d - 1, 2n - 2^d of them at d, so the mean is d + 1 - 2^d / n."""
    d = depth_of(n)
    return 0.0 if n <= 1 else d + 1 - 2 ** d / n


def proof_bytes(n: int) -> int:
    """What a reader receives to check one block: the path and the stored leaf."""
    return HASH_LEN * depth_of(n) + HASH_LEN


def root_from_path(n: int, i: int, leaf: bytes, path: list) -> bytes:
    """The root that a leaf at position i and its sibling path lead to (what the verifier computes)."""
    h = leaf
    for sibling, side in zip(path, reversed(sides_of(n, i))):
        h = node_hash(h, sibling) if side == "L" else node_hash(sibling, h)
    return h


class PathUpdate:
    """The verifier's check of one update, the Wang way; .hashes counts its hash evaluations."""

    def __init__(self):
        self.hashes = 0

    def check(self, root: bytes, n: int, i: int, old_leaf: bytes, new_leaf: bytes, path: list) -> bytes:
        """Verify the old leaf against the root along its path, then return the new root along the
        same path.  Raises ValueError if the old leaf does not lead to the root."""
        if len(path) != len(sides_of(n, i)):
            raise ValueError("path length does not fit the position")
        if root_from_path(n, i, old_leaf, path) != root:
            raise ValueError("the old block does not lead to the root")
        self.hashes += len(path)
        new_root = root_from_path(n, i, new_leaf, path)
        self.hashes += len(path)
        return new_root


def access_costs(n: int, reads: int, writes: int) -> dict:
    """Per-workload totals when every access lands in the one tree of n blocks: the verifier
    spends two paths per update and the server rehashes one; a read carries one path, a path
    being the mean over the n positions (about log2 n), as the measured rows count it.  The same
    dict shape as workload.py's configurations."""
    d = mean_path(n)
    return dict(owner_hashes=writes * 2 * d, provider_leaves=writes * d, read_path=reads * d,
                reads=reads, writes=writes, owner_ms=None, provider_ms=None, splits=0, merges=0,
                trees=1, mean_tree=n)


def fetch_all_bytes(n: int) -> int:
    """Localization in a scheme without one: every leaf hash of the tree."""
    return HASH_LEN * n


def fetch_all(prov, own: bytes, tree_id: bytes, expected_leaves: list) -> tuple[list, int]:
    """Find the bad blocks by fetching every leaf hash the provider recomputes from what it holds
    and comparing with what the verifier expects.  Returns (positions that differ, bytes fetched).
    Trustworthy only against a semi-honest provider, as any fetch of hashes is."""
    leaves = prov.recomputed_leaves(own, tree_id)
    if len(leaves) != len(expected_leaves):
        return list(range(len(expected_leaves))), HASH_LEN * len(leaves)
    return [i for i, (a, b) in enumerate(zip(leaves, expected_leaves)) if a != b], HASH_LEN * len(leaves)
