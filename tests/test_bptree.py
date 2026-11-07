"""The B+ tree of Section III-B: find, predecessor, rank, select, ranges, and bulk load, checked
against a sorted list under random inserts and deletes at small fanouts, where nodes split and
merge all the time."""
import random
from bisect import bisect_left, bisect_right

import pytest

from amf.common.bptree import BPlusTree


def oracle_pred(keys, k):
    i = bisect_right(keys, k)
    return keys[i - 1] if i else None


@pytest.mark.parametrize("fanout", [4, 5, 8, 64])
def test_tree_agrees_with_a_sorted_list_under_random_updates(fanout):
    rnd = random.Random(fanout)
    t, keys = BPlusTree(fanout), []
    for step in range(3000):
        k = rnd.randrange(500)
        if k in keys and rnd.random() < 0.45:
            assert t.delete(k)
            keys.remove(k)
        else:
            new = k not in keys
            assert t.insert(k, ("v", k, step)) is new
            if new:
                keys.append(k)
                keys.sort()
        if step % 97 == 0:
            t.check()
            assert len(t) == len(keys) and list(t) == keys
            for q in (rnd.randrange(-5, 505) for _ in range(20)):
                assert t.rank(q) == bisect_left(keys, q)
                p = t.predecessor(q)
                assert (p[0] if p else None) == oracle_pred(keys, q)
                assert t.get(q) == (("v", q, t.get(q)[2]) if q in keys else None)
            for r in (rnd.randrange(len(keys)) for _ in range(10) if keys):
                assert t.select(r)[0] == keys[r]
            lo, hi = sorted(rnd.sample(range(500), 2))
            assert [k for k, _ in t.items(lo, hi)] == keys[bisect_left(keys, lo):bisect_left(keys, hi)]
            assert t.count(lo, hi) == bisect_left(keys, hi) - bisect_left(keys, lo)
    assert not t.delete(10**9)


def test_insert_replaces_a_value_without_changing_the_count():
    t = BPlusTree(4)
    assert t.insert(7, "a") and not t.insert(7, "b")
    assert len(t) == 1 and t.get(7) == "b"


def test_predecessor_crosses_leaves_and_is_none_below_the_first_key():
    t = BPlusTree(4)
    for k in range(10, 100, 10):
        t.insert(k, k)
    assert t.predecessor(5) is None
    assert t.predecessor(10) == (10, 10) and t.predecessor(19) == (10, 10)
    assert t.predecessor(55) == (50, 50) and t.predecessor(1000) == (90, 90)
    for k in range(40, 70, 10):                   # empty the middle, the links still carry the search
        t.delete(k)
    assert t.predecessor(65) == (30, 30)


def test_select_and_rank_are_inverse():
    t = BPlusTree(5)
    keys = sorted(random.Random(3).sample(range(10**6), 2000))
    for k in keys:
        t.insert(k, None)
    for r in range(0, 2000, 37):
        k, _ = t.select(r)
        assert k == keys[r] and t.rank(k) == r
    with pytest.raises(IndexError):
        t.select(2000)


@pytest.mark.parametrize("n", [0, 1, 3, 4, 5, 17, 64, 65, 1000, 4097])
def test_bulk_load_equals_incremental_inserts(n):
    pairs = [(k * 3, k) for k in range(n)]
    for fanout in (4, 64):
        fast = BPlusTree.bulk_load(pairs, fanout)
        fast.check()
        slow = BPlusTree(fanout)
        for k, v in pairs:
            slow.insert(k, v)
        assert list(fast.items()) == list(slow.items()) == pairs
        assert all(fast.rank(k) == slow.rank(k) for k, _ in pairs)
        fast.insert(-1, "front")
        fast.delete(0)
        fast.check()
        assert fast.predecessor(0) == (-1, "front")


def test_depth_grows_as_the_logarithm_of_the_size():
    def depth(t):
        node, d = t._root, 1
        while hasattr(node, "children"):
            node, d = node.children[0], d + 1
        return d
    assert depth(BPlusTree.bulk_load(((k, None) for k in range(4_000)), 64)) == 2      # 64^2 > 4,000
    assert depth(BPlusTree.bulk_load(((k, None) for k in range(100_000)), 64)) == 3   # 64^3 > 100,000
    assert depth(BPlusTree.bulk_load(((k, None) for k in range(300_000)), 64)) == 4   # 64^3 < 300,000