"""Merkle tree of the canonical shape -- Definition 1 and Lemma 1.

A node that covers m > 1 leaves has the first ceil(m/2) of them in its left
subtree and the rest in its right one.  Consequences the paper relies on:

* the shape depends on n alone and the depth is ceil(log2 n)
  -> ten rounds of Phase 6 for a tree of N_up = 1000 leaves
* two trees over the same leaf values in the same order have the same root,
  however each was built, so the owner can compute any root from its catalog
* if n_b >= 1 and n_a = ceil((n_a + n_b)/2), root(all) = H(r_a || r_b): a split
  at the median, or a merge of two trees of nearly equal size, is checked
  with one hash

Hashing: a leaf value is H(id || CT) (crypto.leaf_hash); an internal node is
H(0x01 || left || right).  Nodes are kept in one array indexed as an implicit
binary tree: the root is index 1, the children of u are 2u and 2u+1.  That is
the provider's Merkle Tree Index for one tree.
"""
import hashlib

HASH_LEN = 32
NODE_PREFIX = b"\x01"
EMPTY_ROOT = hashlib.sha256(b"\x00").digest()      # root of a tree with no leaves


def node_hash(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def depth_of(n: int) -> int:
    """ceil(log2 n) for n >= 1; 0 for the empty tree."""
    return 0 if n <= 1 else (n - 1).bit_length()


def left_size(m: int) -> int:
    """How many of m leaves go to the left child: ceil(m/2)."""
    return (m + 1) // 2


def sides_of(n: int, i: int) -> list[str]:
    """Top-down list of 'L'/'R': which child position i lies in at each level.

    This is all a verifier needs to know about the tree's shape, and it follows
    from (n, i) alone -- so a path proves the position as well as the leaf.
    """
    if not 0 <= i < n:
        raise ValueError("position out of range")
    sides, lo, hi = [], 0, n
    while hi - lo > 1:
        mid = lo + left_size(hi - lo)
        if i < mid:
            sides.append("L")
            hi = mid
        else:
            sides.append("R")
            lo = mid
    return sides


class MerkleTree:
    def __init__(self, leaves: list[bytes]):
        self.n = len(leaves)
        self.depth = depth_of(self.n)
        self._nodes = bytearray(HASH_LEN * (2 << self.depth))   # index 0 unused
        if self.n:
            self._build(1, 0, self.n, leaves)

    # -- construction -------------------------------------------------------

    def _build(self, k: int, lo: int, hi: int, leaves: list[bytes]) -> bytes:
        if hi - lo == 1:
            h = leaves[lo]
        else:
            mid = lo + left_size(hi - lo)
            h = node_hash(self._build(2 * k, lo, mid, leaves),
                          self._build(2 * k + 1, mid, hi, leaves))
        self._set(k, h)
        return h

    def _get(self, k: int) -> bytes:
        return bytes(self._nodes[HASH_LEN * k: HASH_LEN * (k + 1)])

    def _set(self, k: int, h: bytes) -> None:
        self._nodes[HASH_LEN * k: HASH_LEN * (k + 1)] = h

    # -- what the index row and the proofs use ------------------------------

    @property
    def root(self) -> bytes:
        return EMPTY_ROOT if self.n == 0 else self._get(1)

    def node(self, k: int) -> bytes:
        """The stored hash at heap index k (1 = root, children of k are 2k and 2k+1)."""
        return self._get(k)

    def children(self) -> tuple[bytes, bytes]:
        """Roots of the two halves (the first round of Phase 6)."""
        if self.n < 2:
            raise ValueError("a tree with fewer than two leaves has no halves")
        return self._get(2), self._get(3)

    def _index_of(self, i: int) -> int:
        """Heap index of the node holding leaf position i."""
        k = 1
        for side in sides_of(self.n, i):
            k = 2 * k if side == "L" else 2 * k + 1
        return k

    def leaf(self, i: int) -> bytes:
        return self._get(self._index_of(i))

    def path(self, i: int) -> list[bytes]:
        """Sibling hashes from the leaf up to the root ("the read path")."""
        k = self._index_of(i)
        siblings = []
        while k > 1:
            siblings.append(self._get(k ^ 1))      # k^1 flips the last bit: the sibling
            k //= 2
        return siblings

    def update(self, i: int, leaf: bytes) -> int:
        """Replace the leaf at position i and rehash the nodes above it, the shape unchanged.
        Returns the hash evaluations spent: one per node on the path."""
        k = self._index_of(i)
        self._set(k, leaf)
        spent = 0
        while k > 1:
            k //= 2
            self._set(k, node_hash(self._get(2 * k), self._get(2 * k + 1)))
            spent += 1
        return spent


def root_of(leaves: list[bytes]) -> bytes:
    """root(x_0, ..., x_{n-1}): n - 1 hash evaluations."""
    return MerkleTree(list(leaves)).root


def is_hash(x) -> bool:
    return isinstance(x, bytes) and len(x) == HASH_LEN


def verify_path(root, n, i, leaf, path) -> bool:
    """Does `path` carry `leaf` at position `i` of an n-leaf tree up to `root`?  False, never an
    error, for an answer that is not even well formed: the shape is fixed by (n, i), and an
    untrusted party chose every other argument."""
    if type(n) is not int or type(i) is not int or not 0 <= i < n:
        return False
    if not is_hash(root) or not is_hash(leaf) or not isinstance(path, (list, tuple)):
        return False
    sides = sides_of(n, i)
    if len(path) != len(sides) or not all(is_hash(h) for h in path):
        return False
    h = leaf
    for sibling, side in zip(path, reversed(sides)):
        h = node_hash(h, sibling) if side == "L" else node_hash(sibling, h)
    return h == root