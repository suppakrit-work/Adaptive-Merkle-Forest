"""A B+ tree with counts -- Section III-B, the structure behind the Distributed
Index at a DHT node (keyed by lo) and the owner's catalog (keyed by id).

All keys sit in the leaves, in order, and the leaves are linked left to right;
the inner nodes are a roadmap of separator keys (Comer, 1979).  Every node also
carries the number of keys below it, so besides find, insert, delete and the
predecessor search a reader's lookup needs (the largest key not above a
value), the tree answers the three counting questions the catalog asks:

* rank(k)      how many keys are below k       (count of a range = two ranks)
* select(r)    the r-th smallest key            (the identifier at a position)
* items(lo,hi) the keys in [lo, hi), in order   (the leaves to hash to a root)

each in O(log_b N) node visits for fanout b.  Keys are integers; values are
whatever the caller stores.  The tree lives in memory; the fanout is a
parameter, 64 by default, so a million keys are four levels deep.
"""
from bisect import bisect_left, bisect_right


class _Leaf:
    __slots__ = ("keys", "vals", "next", "prev")

    def __init__(self):
        self.keys, self.vals = [], []
        self.next = self.prev = None

    @property
    def size(self) -> int:
        return len(self.keys)


class _Inner:
    __slots__ = ("keys", "children", "size")     # keys[i] = smallest key under children[i + 1]

    def __init__(self, keys, children):
        self.keys, self.children = keys, children
        self.size = sum(c.size for c in children)

    def recount(self) -> None:
        self.size = sum(c.size for c in self.children)


class BPlusTree:
    def __init__(self, fanout: int = 64):
        if fanout < 4:
            raise ValueError("a B+ tree needs a fanout of at least 4")
        self.b = fanout                            # most keys in a leaf, most children of an inner node
        self._min = (fanout + 1) // 2              # least keys in a leaf / children of a non-root inner node
        self._root = _Leaf()
        self._first = self._root
        self._last_was_new = False

    # -- the size, and walking the leaves ----------------------------------------------

    def __len__(self) -> int:
        return self._root.size

    def __contains__(self, key: int) -> bool:
        leaf, i = self._leaf_for(key)
        return i < len(leaf.keys) and leaf.keys[i] == key

    def __iter__(self):
        for k, _ in self.items():
            yield k

    def items(self, lo: int | None = None, hi: int | None = None):
        """(key, value) pairs with lo <= key < hi, in key order, along the leaf links."""
        if lo is None:
            leaf, i = self._first, 0
        else:
            leaf, i = self._leaf_for(lo)
        while leaf is not None:
            for j in range(i, len(leaf.keys)):
                if hi is not None and leaf.keys[j] >= hi:
                    return
                yield leaf.keys[j], leaf.vals[j]
            leaf, i = leaf.next, 0

    # -- find -------------------------------------------------------------------------

    def _leaf_for(self, key: int):
        """The leaf that holds key or would, and the index of the first key there >= key."""
        node = self._root
        while isinstance(node, _Inner):
            node = node.children[bisect_right(node.keys, key)]
        return node, bisect_left(node.keys, key)

    def get(self, key: int, default=None):
        leaf, i = self._leaf_for(key)
        return leaf.vals[i] if i < len(leaf.keys) and leaf.keys[i] == key else default

    def predecessor(self, key: int):
        """(k, value) for the largest k <= key, or None: what routes an identifier to the row
        whose range starts at or before it."""
        leaf, i = self._leaf_for(key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return key, leaf.vals[i]
        if i > 0:
            return leaf.keys[i - 1], leaf.vals[i - 1]
        leaf = leaf.prev
        while leaf is not None and not leaf.keys:
            leaf = leaf.prev
        return (leaf.keys[-1], leaf.vals[-1]) if leaf is not None else None

    def rank(self, key: int) -> int:
        """How many keys are < key: the counts of the subtrees passed on the way down."""
        node, r = self._root, 0
        while isinstance(node, _Inner):
            i = bisect_right(node.keys, key)
            for c in node.children[:i]:
                r += c.size
            node = node.children[i]
        return r + bisect_left(node.keys, key)

    def select(self, r: int):
        """(key, value) of the r-th smallest key, r from 0; IndexError past the end."""
        if not 0 <= r < len(self):
            raise IndexError("rank out of range")
        node = self._root
        while isinstance(node, _Inner):
            for c in node.children:
                if r < c.size:
                    node = c
                    break
                r -= c.size
        return node.keys[r], node.vals[r]

    def count(self, lo: int, hi: int) -> int:
        """How many keys lie in [lo, hi)."""
        return self.rank(hi) - self.rank(lo) if hi > lo else 0

    # -- insert ---------------------------------------------------------------------------

    def insert(self, key: int, value) -> bool:
        """Store value under key; True if the key is new, False if its value was replaced."""
        split = self._insert(self._root, key, value)
        if split is None:
            return self._last_was_new
        sep, right = split
        self._root = _Inner([sep], [self._root, right])
        return self._last_was_new

    def _insert(self, node, key, value):
        """Returns None, or (separator, new right sibling) when node split."""
        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                node.vals[i] = value
                self._last_was_new = False
                return None
            node.keys.insert(i, key)
            node.vals.insert(i, value)
            self._last_was_new = True
            if len(node.keys) <= self.b:
                return None
            right = _Leaf()
            h = len(node.keys) // 2
            right.keys, right.vals = node.keys[h:], node.vals[h:]
            node.keys, node.vals = node.keys[:h], node.vals[:h]
            right.next, right.prev, node.next = node.next, node, right
            if right.next is not None:
                right.next.prev = right
            return right.keys[0], right
        i = bisect_right(node.keys, key)
        split = self._insert(node.children[i], key, value)
        if split is not None:
            sep, right = split
            node.keys.insert(i, sep)
            node.children.insert(i + 1, right)
        node.recount()
        if len(node.children) <= self.b:
            return None
        h = len(node.children) // 2
        right = _Inner(node.keys[h:], node.children[h:])
        sep = node.keys[h - 1]
        node.keys, node.children = node.keys[:h - 1], node.children[:h]
        node.recount()
        return sep, right

    # -- delete ---------------------------------------------------------------------------

    def delete(self, key: int) -> bool:
        """Remove key; False if it was not there."""
        found = self._delete(self._root, key)
        root = self._root
        if isinstance(root, _Inner) and len(root.children) == 1:
            self._root = root.children[0]
        return found

    def _delete(self, node, key) -> bool:
        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i == len(node.keys) or node.keys[i] != key:
                return False
            del node.keys[i], node.vals[i]
            return True
        i = bisect_right(node.keys, key)
        child = node.children[i]
        found = self._delete(child, key)
        if found and self._underfull(child):
            self._rebalance(node, i)
        node.recount()
        return found

    def _underfull(self, node) -> bool:
        if isinstance(node, _Leaf):
            return len(node.keys) < self._min
        return len(node.children) < self._min

    def _rebalance(self, parent, i) -> None:
        """Child i of parent is underfull: borrow from a sibling if it can spare, else merge."""
        child = parent.children[i]
        left = parent.children[i - 1] if i > 0 else None
        right = parent.children[i + 1] if i + 1 < len(parent.children) else None
        if isinstance(child, _Leaf):
            if left is not None and len(left.keys) > self._min:
                child.keys.insert(0, left.keys.pop())
                child.vals.insert(0, left.vals.pop())
                parent.keys[i - 1] = child.keys[0]
            elif right is not None and len(right.keys) > self._min:
                child.keys.append(right.keys.pop(0))
                child.vals.append(right.vals.pop(0))
                parent.keys[i] = right.keys[0]
            elif left is not None:
                left.keys += child.keys
                left.vals += child.vals
                left.next = child.next
                if child.next is not None:
                    child.next.prev = left
                del parent.keys[i - 1], parent.children[i]
            else:
                child.keys += right.keys
                child.vals += right.vals
                child.next = right.next
                if right.next is not None:
                    right.next.prev = child
                del parent.keys[i], parent.children[i + 1]
            return
        if left is not None and len(left.children) > self._min:
            child.children.insert(0, left.children.pop())
            child.keys.insert(0, parent.keys[i - 1])
            parent.keys[i - 1] = left.keys.pop()
            left.recount()
            child.recount()
        elif right is not None and len(right.children) > self._min:
            child.children.append(right.children.pop(0))
            child.keys.append(parent.keys[i])
            parent.keys[i] = right.keys.pop(0)
            right.recount()
            child.recount()
        elif left is not None:
            left.keys += [parent.keys[i - 1]] + child.keys
            left.children += child.children
            left.recount()
            del parent.keys[i - 1], parent.children[i]
        else:
            child.keys += [parent.keys[i]] + right.keys
            child.children += right.children
            child.recount()
            del parent.keys[i], parent.children[i + 1]

    # -- bulk load ---------------------------------------------------------------------------

    @classmethod
    def bulk_load(cls, pairs, fanout: int = 64) -> "BPlusTree":
        """Build from (key, value) pairs sorted by key with no repeats, in O(N): leaves filled
        to the fanout, then one level of inner nodes at a time."""
        t = cls(fanout)
        pairs = list(pairs)
        if not pairs:
            return t
        leaves, prev = [], None
        for at in range(0, len(pairs), fanout):
            leaf = _Leaf()
            leaf.keys = [k for k, _ in pairs[at:at + fanout]]
            leaf.vals = [v for _, v in pairs[at:at + fanout]]
            leaf.prev = prev
            if prev is not None:
                prev.next = leaf
            leaves.append(leaf)
            prev = leaf
        if len(leaves) > 1 and len(leaves[-1].keys) < t._min:       # the last leaf takes from its left
            a, b = leaves[-2], leaves[-1]
            move = t._min - len(b.keys)
            b.keys, b.vals = a.keys[-move:] + b.keys, a.vals[-move:] + b.vals
            a.keys, a.vals = a.keys[:-move], a.vals[:-move]
        level = leaves
        while len(level) > 1:
            parents = []
            for at in range(0, len(level), fanout):
                group = level[at:at + fanout]
                parents.append(_Inner([t._first_key(c) for c in group[1:]], group))
            if len(parents) > 1 and len(parents[-1].children) < t._min:
                a, b = parents[-2], parents[-1]
                move = t._min - len(b.children)
                moved = a.children[-move:]
                b.children = moved + b.children
                b.keys = [t._first_key(c) for c in b.children[1:]]
                a.children = a.children[:-move]
                a.keys = a.keys[:len(a.children) - 1]
                a.recount()
                b.recount()
            level = parents
        t._root, t._first = level[0], leaves[0]
        return t

    @staticmethod
    def _first_key(node) -> int:
        while isinstance(node, _Inner):
            node = node.children[0]
        return node.keys[0]

    # -- for tests ----------------------------------------------------------------------------

    def check(self) -> None:
        """Raise AssertionError if an invariant is broken (tests only)."""
        def walk(node, lo, hi, depth):
            if isinstance(node, _Leaf):
                assert node.keys == sorted(node.keys) and len(node.keys) == len(node.vals)
                assert all((lo is None or lo <= k) and (hi is None or k < hi) for k in node.keys)
                assert node is self._root or len(node.keys) >= self._min
                assert len(node.keys) <= self.b
                return depth, len(node.keys)
            assert len(node.children) == len(node.keys) + 1
            assert node is self._root or len(node.children) >= self._min
            assert len(node.children) <= self.b
            bounds = [lo] + node.keys + [hi]
            depths, total = set(), 0
            for c, l, h in zip(node.children, bounds, bounds[1:]):
                d, n = walk(c, l, h, depth + 1)
                depths.add(d)
                total += n
            assert len(depths) == 1 and total == node.size
            return depths.pop(), total
        walk(self._root, None, None, 0)
        keys = [k for k, _ in self.items()]
        assert keys == sorted(keys) and len(keys) == len(self)