"""HMT, the Huffman-Merkle Tree of Shangguan, Yaish and Malkhi (ePrint 2026/1235): the
frequency-aware baseline, run on the same objects and access sequences as AMF.

Z. Shangguan, A. Yaish and D. Malkhi, "Authenticated Data Structures for
Dynamic Workloads", Cryptology ePrint Archive 2026/1235, keep hot elements in
a Huffman-shaped Merkle tree (HuffMHT) and cold ones in a balanced binary
Merkle tree; one global root H(r_cold || r_hot) commits to both tiers
(Sec. 4, Algorithm 3).  Access frequencies are estimated with a Count-Min
Sketch (Sec. 4.2); a bounded Promotion Cache holds the candidates for
promotion and the exact membership of the hot tier (Sec. 4.3); at period
boundaries a policy promotes and demotes between tiers (Sec. 5; here the
Ratio-Based policy of Sec. 5.2 and Algorithm 5: a candidate is promoted when
its lifetime access rate f/B exceeds theta, into spare capacity or in place of
a strictly colder hot element).  Elements promoted between rebuilds live in
an overflow Merkle tree; every R periods the hot tier is rebuilt from the
current frequencies and the overflow tree is emptied (periodic-rebuild
HuffMHT, Sec. 3).  Value updates rehash the dirty path of their tier and the
tier and global roots.

Taken from the paper: the two tiers, the HuffMHT layout, the periodic rebuild
with an overflow tree, the global root, the CMS with conservative update, the
capacity-bounded promotion cache and the Ratio-Based policy.  Left out: the
Sliding-Window and Dynamic-Control policies, the bucketed LRU details of the
cache (candidates are ranked by their CMS estimate directly) and BLAKE3 (the
CMS rows are derived from SHA-256).  The paper's cold tier is "a balanced
binary MT"; here the cold tier and the overflow tree are binary hash tries
keyed by a 32-byte hash of the element (each key sits at the first depth at
which it is alone, as in a binary trie such as Ethereum's UBT), which gives
the logarithmic insert, delete and update paths the paper asks of the cold
tier at the price of paths about one hash longer than a balanced tree's
(mean 12.3 at n = 2,000 against ceil(log2 n) = 11).  The keys must be
uniform: AMF identifiers share their first 33 bytes, so workload.py keys the
tries by SHA-256 values, not by AMF identifiers.  Hashing is AMF's (SHA-256, the same leaf and node
encodings), so the two schemes differ only in layout and in who can check
what.

What the comparison measures, per access and on the same stream as AMF, all
by running the code and counting its hash evaluations (the paper counts hashed
bytes; one evaluation here hashes 65 bytes): hashes at the maintainer (the
provider: dirty paths, tier moves, rebuilds), hashes at a Verifier that checks
reads by their proof and rewrites by the path update (Wang's way, not in HMT)
and TRUSTS the layout changes, which is the paper's model (a client checks
proofs against committed roots), hashes at a Verifier of our own that instead
recomputes from their leaves the tiers whose root changed (Algorithm 3 gives a
verifier no proof for a tier move or a rebuild), and the proof length of a
read in hashes (the paper's weighted proof size, in hashes rather than bytes).
"""
import hashlib
import heapq
from collections import Counter

from amf.common.merkle import EMPTY_ROOT, node_hash

TIER_PREFIX = b"\x02"            # H(0x02 || tag || r_base [|| r_overflow]) for the hot tier (Sec. 3)
GLOBAL_PREFIX = b"\x03"          # H(0x03 || r_cold || r_hot) (Sec. 4.1)
BASE_ONLY, BASE_OVERFLOW = b"\x00", b"\x01"


class HashCount:
    """One counter per party; every node hash a structure computes is charged to its owner."""

    def __init__(self):
        self.n = 0

    def node(self, left, right):
        self.n += 1
        return node_hash(left, right)

    def tier(self, *parts):
        self.n += 1
        return hashlib.sha256(b"".join(parts)).digest()


# ---- the cold tier and the overflow tree: a binary hash trie ---------------------------------

def _bit(key: bytes, depth: int) -> int:
    return (key[depth >> 3] >> (7 - (depth & 7))) & 1


class _TrieLeaf:
    __slots__ = ("key", "value", "hash")

    def __init__(self, key, value):
        self.key, self.value, self.hash = key, value, value


class _TrieBranch:
    __slots__ = ("left", "right", "hash")

    def __init__(self, left=None, right=None):
        self.left, self.right, self.hash = left, right, None


class BinaryTrie:
    """Merkle tree over keys as a binary trie: a key lives at the first depth where it is alone,
    so every path is about log2 n long and an insert, delete or update rehashes one path."""

    def __init__(self, counter: HashCount):
        self.c = counter
        self.root = None
        self.size = 0

    # -- hashing --------------------------------------------------------------------------
    def _hash(self, node) -> bytes:
        return EMPTY_ROOT if node is None else node.hash

    def _rehash(self, node) -> None:
        node.hash = self.c.node(self._hash(node.left), self._hash(node.right))

    @property
    def root_hash(self) -> bytes:
        return self._hash(self.root)

    @classmethod
    def build(cls, counter: HashCount, items) -> "BinaryTrie":
        """The trie over (key, value) pairs in one pass: one hash per branch node."""
        t = cls(counter)
        items = sorted(items)
        t.size = len(items)

        def make(part, depth):
            if len(part) == 1:
                return _TrieLeaf(*part[0])
            cut = 0
            while cut < len(part) and not _bit(part[cut][0], depth):
                cut += 1
            node = _TrieBranch(make(part[:cut], depth + 1) if cut else None,
                               make(part[cut:], depth + 1) if cut < len(part) else None)
            t._rehash(node)
            return node
        t.root = make(items, 0) if items else None
        return t

    # -- find --------------------------------------------------------------------------------
    def _descend(self, key):
        """(nodes on the path from the root, the leaf or None)."""
        path, node, depth = [], self.root, 0
        while isinstance(node, _TrieBranch):
            path.append(node)
            node = node.right if _bit(key, depth) else node.left
            depth += 1
        return path, node

    def get(self, key):
        _, node = self._descend(key)
        return node.value if isinstance(node, _TrieLeaf) and node.key == key else None

    def depth_of(self, key) -> int:
        path, node = self._descend(key)
        if not (isinstance(node, _TrieLeaf) and node.key == key):
            raise KeyError("not in this tree")
        return len(path)

    # -- change ------------------------------------------------------------------------------
    def put(self, key: bytes, value: bytes) -> int:
        """Insert or update; returns the hashes spent (the dirty path)."""
        before = self.c.n
        if self.root is None:
            self.root = _TrieLeaf(key, value)
            self.size = 1
            return 0
        path, node = self._descend(key)
        depth = len(path)
        if isinstance(node, _TrieLeaf) and node.key == key:
            node.value = node.hash = value
        else:
            new = _TrieLeaf(key, value)
            self.size += 1
            if node is None:
                child = new
            else:                                   # push the resident leaf down until the bits differ
                child = branch = _TrieBranch()
                d = depth
                while _bit(key, d) == _bit(node.key, d):
                    nxt = _TrieBranch()
                    if _bit(key, d):
                        branch.right = nxt
                    else:
                        branch.left = nxt
                    branch, d = nxt, d + 1
                if _bit(key, d):
                    branch.right, branch.left = new, node
                else:
                    branch.left, branch.right = new, node
                self._rehash_down(child)
            if not path:
                self.root = child
            elif _bit(key, depth - 1):
                path[-1].right = child
            else:
                path[-1].left = child
        for b in reversed(path):
            self._rehash(b)
        return self.c.n - before

    def _rehash_down(self, node):
        if isinstance(node, _TrieBranch):
            self._rehash_down(node.left)
            self._rehash_down(node.right)
            self._rehash(node)

    def delete(self, key: bytes) -> int:
        """Remove a key; a branch left with a single leaf collapses into it.  Returns the hashes spent."""
        before = self.c.n
        path, node = self._descend(key)
        if not (isinstance(node, _TrieLeaf) and node.key == key):
            raise KeyError("not in this tree")
        self.size -= 1
        if not path:
            self.root = None
            return 0
        current = path.pop()
        if current.left is node:
            current.left = None
        else:
            current.right = None
        while True:
            left, right = current.left, current.right
            if left is None and right is None:
                replacement = None
            elif left is None and isinstance(right, _TrieLeaf):
                replacement = right
            elif right is None and isinstance(left, _TrieLeaf):
                replacement = left
            else:
                break                                   # still a proper branch
            if not path:
                self.root = replacement
                return self.c.n - before
            parent = path.pop()
            if parent.left is current:
                parent.left = replacement
            else:
                parent.right = replacement
            current = parent
        self._rehash(current)
        for b in reversed(path):
            self._rehash(b)
        return self.c.n - before

    # -- proofs ---------------------------------------------------------------------------------
    def proof(self, key: bytes) -> list:
        """Sibling hashes from the leaf up to the root, with the side of the key at each level."""
        path, node = self._descend(key)
        if not (isinstance(node, _TrieLeaf) and node.key == key):
            raise KeyError("not in this tree")
        out = []
        for depth, branch in enumerate(path):
            if _bit(key, depth):
                out.append(("R", self._hash(branch.left)))
            else:
                out.append(("L", self._hash(branch.right)))
        return list(reversed(out))

    def keys(self):
        out, stack = [], [self.root]
        while stack:
            node = stack.pop()
            if isinstance(node, _TrieLeaf):
                out.append((node.key, node.value))
            elif isinstance(node, _TrieBranch):
                stack.append(node.left)
                stack.append(node.right)
        return out


def verify_proof(root: bytes, leaf: bytes, proof: list, counter: HashCount) -> bool:
    """Recompute the root from a leaf and its (side, sibling) proof -- HMT Algorithm 1."""
    h = leaf
    for side, sibling in proof:
        h = counter.node(h, sibling) if side == "L" else counter.node(sibling, h)
    return h == root


# ---- the hot tier: a HuffMHT ------------------------------------------------------------------

class _HuffNode:
    __slots__ = ("left", "right", "parent", "hash", "key")

    def __init__(self, key=None, hash=None):
        self.left = self.right = self.parent = None
        self.hash, self.key = hash, key


class HuffMHT:
    """Huffman-shaped Merkle tree over (key, leaf, weight): the heavier the key, the shorter
    its path (Sec. 2.4).  Built in one go; a value update rehashes the path to the root."""

    def __init__(self, counter: HashCount, items):
        self.c = counter
        self.leaves = {}
        self.root = None
        items = list(items)
        self.built_with = [(k, w) for k, _, w in items]      # the layout, as the maintainer publishes it
        if not items:
            return
        heap, order = [], 0
        for key, leaf, weight in items:
            n = _HuffNode(key, leaf)
            self.leaves[key] = n
            heap.append((weight, order, n))
            order += 1
        heapq.heapify(heap)
        while len(heap) > 1:
            wa, _, a = heapq.heappop(heap)
            wb, _, b = heapq.heappop(heap)
            p = _HuffNode()
            p.left, p.right, a.parent, b.parent = a, b, p, p
            p.hash = self.c.node(a.hash, b.hash)
            heapq.heappush(heap, (wa + wb, order, p))
            order += 1
        self.root = heap[0][2]

    @property
    def root_hash(self) -> bytes:
        return EMPTY_ROOT if self.root is None else self.root.hash

    def __contains__(self, key):
        return key in self.leaves

    def __len__(self):
        return len(self.leaves)

    def depth_of(self, key) -> int:
        n, d = self.leaves[key], 0
        while n.parent is not None:
            n, d = n.parent, d + 1
        return d

    def update(self, key, leaf) -> int:
        """Replace a leaf value and rehash its path; returns the hashes spent."""
        before, n = self.c.n, self.leaves[key]
        n.hash = leaf
        while n.parent is not None:
            n = n.parent
            n.hash = self.c.node(n.left.hash, n.right.hash)
        return self.c.n - before

    def proof(self, key) -> list:
        out, n = [], self.leaves[key]
        while n.parent is not None:
            p = n.parent
            out.append(("L", p.right.hash) if p.left is n else ("R", p.left.hash))
            n = p
        return out

    def items(self):
        return [(k, n.hash) for k, n in self.leaves.items()]


# ---- frequency tracking and the promotion cache ----------------------------------------------

class CountMinSketch:
    """d rows of w counters, rows from SHA-256 of the key, conservative update (Sec. 4.2)."""

    def __init__(self, rows: int = 4, width: int = 1 << 14):
        self.d, self.w = rows, width
        self.table = [[0] * width for _ in range(rows)]
        self.total = 0

    def _cells(self, key):
        h = hashlib.sha256(key).digest()
        return [(r, int.from_bytes(h[4 * r:4 * r + 4], "big") % self.w) for r in range(self.d)]

    def add(self, key, by: int = 1) -> None:
        cells = self._cells(key)
        est = min(self.table[r][i] for r, i in cells)
        for r, i in cells:
            self.table[r][i] = max(self.table[r][i], est + by)
        self.total += by

    def estimate(self, key) -> int:
        return min(self.table[r][i] for r, i in self._cells(key))


class PromotionCache:
    """The bounded set of keys a tier's policy can see (Sec. 4.3): for the cold tier the
    candidates for promotion, for the hot tier its exact membership.  Ranked by CMS estimate."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = {}                 # key -> last estimate seen

    def touch(self, key, estimate) -> None:
        self.keys[key] = estimate
        if len(self.keys) > self.capacity:
            coldest = min(self.keys, key=self.keys.get)
            del self.keys[coldest]

    def ranked(self, cms):
        """Keys from the most to the least frequent by the sketch's current estimate."""
        return sorted(self.keys, key=lambda k: -cms.estimate(k))

    def discard(self, key) -> None:
        self.keys.pop(key, None)


# ---- HMT ---------------------------------------------------------------------------------------

class HMT:
    """Two tiers, one global root, the Ratio-Based policy, periodic rebuilds.

    theta: lifetime access rate (accesses per period) a cold candidate needs to be promoted.
    hot_capacity: most elements in the hot tier (base HuffMHT + overflow).
    candidates: how many cold keys the promotion cache follows.
    rebuild_every: periods between two rebuilds of the HuffMHT.
    """

    def __init__(self, items, theta: float = 0.5, hot_capacity: int = 1000, candidates: int = 2000,
                 rebuild_every: int = 5):
        self.maintainer = HashCount()
        self.theta, self.hot_capacity, self.rebuild_every = theta, hot_capacity, rebuild_every
        self.cold = BinaryTrie.build(self.maintainer, items)
        self.base = HuffMHT(self.maintainer, [])
        self.overflow = BinaryTrie(self.maintainer)
        self.cms = CountMinSketch()
        self.cold_cache = PromotionCache(candidates)
        self.hot_cache = PromotionCache(hot_capacity)
        self.periods = 0
        self.stats = Counter()
        self.global_root = None
        self._commit()

    # -- where a key lives -----------------------------------------------------------------------
    def tier_of(self, key) -> str:
        if key in self.base:
            return "base"
        if self.overflow.get(key) is not None:
            return "overflow"
        return "cold"

    def hot_root(self) -> bytes:
        return hot_root_from(self.base.root_hash, self.overflow.root_hash, self.overflow.size > 0, self.maintainer)

    def _commit(self) -> None:
        self.global_root = self.maintainer.tier(GLOBAL_PREFIX, self.cold.root_hash, self.hot_root())

    # -- accesses ---------------------------------------------------------------------------------
    def _track(self, key) -> None:
        self.cms.add(key)
        if self.tier_of(key) == "cold":
            self.cold_cache.touch(key, self.cms.estimate(key))
        else:
            self.hot_cache.touch(key, self.cms.estimate(key))

    def roots(self) -> tuple:
        """What a proof carries beside the path: the tier roots (Algorithm 3, GetProof)."""
        return (self.cold.root_hash, self.base.root_hash, self.overflow.root_hash, self.overflow.size > 0)

    def read(self, key, track: bool = True) -> tuple:
        """(proof as (side, sibling) list, tier, tier roots) for a verifier holding the global root;
        the proof length is what Sec. 6 calls the membership-proof size.  track=False takes the proof
        without counting an access (a rewrite checked on its path is counted once, by write)."""
        if track:
            self._track(key)
        tier = self.tier_of(key)
        proof = {"cold": self.cold.proof, "base": self.base.proof, "overflow": self.overflow.proof}[tier](key)
        return proof, tier, self.roots()

    def write(self, key, leaf) -> int:
        """A value update: rehash the dirty path, the tier root and the global root.
        Returns the maintainer's hashes."""
        before = self.maintainer.n
        self._track(key)
        tier = self.tier_of(key)
        if tier == "cold":
            self.cold.put(key, leaf)
        elif tier == "base":
            self.base.update(key, leaf)
        else:
            self.overflow.put(key, leaf)
        self._commit()
        self.stats["writes"] += 1
        return self.maintainer.n - before

    def proof_length(self, key) -> int:
        """Hashes in a membership proof: the path in its tier, the other tier's root, and the
        other component of the hot tier when there is one."""
        tier = self.tier_of(key)
        if tier == "cold":
            return self.cold.depth_of(key) + 1
        extra = 2 if self.overflow.size else 1
        return (self.base.depth_of(key) if tier == "base" else self.overflow.depth_of(key)) + extra

    # -- the policy, at a period boundary ---------------------------------------------------------
    def end_of_period(self) -> dict:
        """Ratio-Based migration (Algorithm 5 with f/B > theta), then a rebuild every
        rebuild_every periods.  Returns what changed and what it cost the maintainer."""
        before = self.maintainer.n
        self.periods += 1
        need = self.theta * self.periods
        promoted = demoted = 0
        hot = lambda: len(self.base) + self.overflow.size
        for key in self.cold_cache.ranked(self.cms):
            if self.cms.estimate(key) <= need:
                break
            if hot() < self.hot_capacity:
                self._promote(key)
                promoted += 1
                continue
            ranked_hot = self.hot_cache.ranked(self.cms)
            tail = ranked_hot[-1] if ranked_hot else None
            if tail is None or self.cms.estimate(key) <= self.cms.estimate(tail):
                break
            self._demote(tail)
            self._promote(key)
            promoted += 1
            demoted += 1
            break                                       # Algorithm 5, step 2: one swap per period
        rebuilt = False
        if self.periods % self.rebuild_every == 0 and (self.overflow.size or promoted or demoted):
            self._rebuild()
            rebuilt = True
        changed = promoted or demoted or rebuilt
        if changed:
            self._commit()
        self.stats["promoted"] += promoted
        self.stats["demoted"] += demoted
        self.stats["rebuilds"] += rebuilt
        return {"promoted": promoted, "demoted": demoted, "rebuilt": rebuilt,
                "maintainer_hashes": self.maintainer.n - before, "changed": bool(changed),
                "hot": hot(), "cold": self.cold.size}

    def _promote(self, key) -> None:
        leaf = self.cold.get(key)
        self.cold.delete(key)
        self.overflow.put(key, leaf)
        self.cold_cache.discard(key)
        self.hot_cache.touch(key, self.cms.estimate(key))

    def _demote(self, key) -> None:
        if key in self.base:
            leaf = dict(self.base.items())[key]
            self.base = HuffMHT(self.maintainer, [(k, v, self.cms.estimate(k)) for k, v in self.base.items()
                                                  if k != key])
        else:
            leaf = self.overflow.get(key)
            self.overflow.delete(key)
        self.cold.put(key, leaf)
        self.hot_cache.discard(key)
        self.cold_cache.touch(key, self.cms.estimate(key))

    def _rebuild(self) -> None:
        items = self.base.items() + self.overflow.keys()
        self.base = HuffMHT(self.maintainer, [(k, v, self.cms.estimate(k)) for k, v in items])
        self.overflow = BinaryTrie(self.maintainer)

    def tier_sizes(self) -> dict:
        return {"cold": self.cold.size, "base": len(self.base), "overflow": self.overflow.size}


# ---- what a verifier does ------------------------------------------------------------------------

def tier_root_from(leaf: bytes, proof: list, counter: HashCount) -> bytes:
    h = leaf
    for side, sibling in proof:
        h = counter.node(h, sibling) if side == "L" else counter.node(sibling, h)
    return h


def hot_root_from(base_root: bytes, overflow_root: bytes, overflow_nonempty: bool, counter: HashCount) -> bytes:
    if overflow_nonempty:
        return counter.tier(TIER_PREFIX, BASE_OVERFLOW, base_root, overflow_root)
    return counter.tier(TIER_PREFIX, BASE_ONLY, base_root)


class Verifier:
    """Holds the global root; checks reads by their proof and rewrites by the path update
    (old leaf to the old root, new root along the same path, as Wang et al.), and either
    trusts a layout change (takes the new global root as announced) or recomputes the tiers
    whose root changed.  .hashes counts what it spent."""

    def __init__(self, global_root: bytes):
        self.global_root = global_root
        self.count = HashCount()

    @property
    def hashes(self) -> int:
        return self.count.n

    def _global(self, tier, tier_root, roots) -> bytes:
        cold, base, overflow, nonempty = roots
        if tier == "cold":
            cold = tier_root
        elif tier == "base":
            base = tier_root
        else:
            overflow = tier_root
        return self.count.tier(GLOBAL_PREFIX, cold, hot_root_from(base, overflow, nonempty, self.count))

    def check_read(self, leaf, proof, tier, roots) -> bool:
        return self._global(tier, tier_root_from(leaf, proof, self.count), roots) == self.global_root

    def check_write(self, old_leaf, new_leaf, proof, tier, roots) -> bool:
        """The old leaf must lead to the root held; the new global root follows along the same path."""
        if not self.check_read(old_leaf, proof, tier, roots):
            return False
        self.global_root = self._global(tier, tier_root_from(new_leaf, proof, self.count), roots)
        return True

    def trust_layout(self, new_global_root: bytes) -> None:
        """A tier move or a rebuild: nothing to check, the verifier takes the announced root."""
        self.global_root = new_global_root

    def rebuild_layout(self, hmt: "HMT", held: tuple | None = None) -> None:
        """The same, for a verifier that will not trust it: recompute from its leaves every tier whose
        announced root is not the one it held (held: the tier roots it had, None = all; it must hold or
        fetch all leaf values of those tiers), then the roots."""
        c = self.count
        cold, base_r, overflow_r, nonempty = hmt.roots()
        had = held or (None, None, None, None)
        if cold != had[0]:
            cold = BinaryTrie.build(c, hmt.cold.keys()).root_hash
        if base_r != had[1]:
            values = dict(hmt.base.items())
            base_r = HuffMHT(c, [(k, values[k], w) for k, w in hmt.base.built_with]).root_hash   # the published weights
        if overflow_r != had[2]:
            overflow_r = BinaryTrie.build(c, hmt.overflow.keys()).root_hash
        self.global_root = c.tier(GLOBAL_PREFIX, cold, hot_root_from(base_r, overflow_r, nonempty, c))
        assert self.global_root == hmt.global_root
