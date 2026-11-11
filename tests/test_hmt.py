"""Tests for the HMT baseline (baselines/shangguan2026.py): the trie, the HuffMHT, the sketch,
the tiers, the policy, and a verifier that checks every answer against the global root."""
import hashlib
import random

import pytest

from baselines.shangguan2026 import (HMT, BinaryTrie, CountMinSketch, HashCount, HuffMHT, Verifier, _TrieBranch,
                                     _TrieLeaf, tier_root_from)


def keyed(n, seed=1):
    rnd = random.Random(seed)
    return [(hashlib.sha256(b"k%d-%d" % (seed, i)).digest(), rnd.randbytes(32)) for i in range(n)]


def no_dangling_leaf(t):
    """A branch never holds a single child that is a leaf (the key would sit one level up)."""
    stack = [t.root]
    while stack:
        n = stack.pop()
        if isinstance(n, _TrieBranch):
            kids = [n.left, n.right]
            assert not (sum(k is not None for k in kids) == 1 and any(isinstance(k, _TrieLeaf) for k in kids))
            stack += [k for k in kids if k is not None]


# ---- the binary trie --------------------------------------------------------------------------

def test_trie_insert_delete_update_keep_proofs_valid():
    rnd, c = random.Random(2), HashCount()
    t, ref = BinaryTrie(c), {}
    for step in range(1500):
        if ref and rnd.random() < 0.4:
            k = rnd.choice(list(ref))
            t.delete(k)
            del ref[k]
        else:
            k, v = hashlib.sha256(rnd.randbytes(8)).digest(), rnd.randbytes(32)
            t.put(k, v)
            ref[k] = v
        if step % 300 == 0:
            assert t.size == len(ref)
            for k, v in ref.items():
                assert t.get(k) == v and tier_root_from(v, t.proof(k), HashCount()) == t.root_hash
            no_dangling_leaf(t)
    with pytest.raises(KeyError):
        t.delete(b"\x00" * 32)


def test_trie_bulk_build_equals_incremental_and_costs_one_hash_per_branch():
    items = keyed(2000)
    c1, c2 = HashCount(), HashCount()
    a = BinaryTrie.build(c1, items)
    b = BinaryTrie(c2)
    for k, v in items:
        b.put(k, v)
    assert a.root_hash == b.root_hash and a.size == b.size == 2000
    assert c1.n < 2 * 2000 < c2.n                      # bulk: about n hashes; one by one: about n log n
    depths = [a.depth_of(k) for k, _ in items]
    assert 9 <= sum(depths) / len(depths) <= 14          # log2 2000 = 11, give or take


def test_trie_put_counts_the_path():
    c = HashCount()
    t = BinaryTrie.build(c, keyed(500))
    k, _ = keyed(500)[7]
    spent = t.put(k, b"\x01" * 32)
    assert spent == t.depth_of(k)


# ---- the HuffMHT ------------------------------------------------------------------------------

def test_huffmht_puts_heavy_keys_near_the_root_and_proves_every_leaf():
    items = [(k, v, 10_000 // (i + 1)) for i, (k, v) in enumerate(keyed(300))]
    c = HashCount()
    h = HuffMHT(c, items)
    assert c.n == 299                                    # one hash per internal node
    for k, v, _ in items:
        assert tier_root_from(v, h.proof(k), HashCount()) == h.root_hash
    assert h.depth_of(items[0][0]) < h.depth_of(items[150][0]) <= h.depth_of(items[-1][0])
    k, _, _ = items[20]
    spent = h.update(k, b"\x02" * 32)
    assert spent == h.depth_of(k) and tier_root_from(b"\x02" * 32, h.proof(k), HashCount()) == h.root_hash


# ---- the sketch --------------------------------------------------------------------------------

def test_count_min_sketch_never_undercounts():
    rnd, cms, truth = random.Random(3), CountMinSketch(rows=4, width=256), {}
    for _ in range(5000):
        k = b"key-%d" % rnd.randrange(400)
        cms.add(k)
        truth[k] = truth.get(k, 0) + 1
    assert all(cms.estimate(k) >= n for k, n in truth.items())
    assert sum(cms.estimate(k) for k in truth) < 2 * sum(truth.values())     # crowded, yet not wildly over


# ---- HMT with a verifier ------------------------------------------------------------------------

def test_hmt_reads_writes_and_layout_changes_verify_against_the_global_root():
    rnd = random.Random(4)
    items = keyed(600)
    leaves = dict(items)
    h = HMT(items, theta=0.5, hot_capacity=60, candidates=120, rebuild_every=2)
    trusting, rebuilding = Verifier(h.global_root), Verifier(h.global_root)
    ranked = [k for k, _ in items]
    rnd.shuffle(ranked)
    weights = [1.0 / (i + 1) for i in range(len(ranked))]
    for step in range(1, 2001):
        k = rnd.choices(ranked, weights)[0]
        if rnd.random() < 0.3:
            proof, tier, roots = h.read(k)
            new = rnd.randbytes(32)
            h.write(k, new)
            assert trusting.check_write(leaves[k], new, proof, tier, roots)
            assert rebuilding.check_write(leaves[k], new, proof, tier, roots)
            leaves[k] = new
        else:
            proof, tier, roots = h.read(k)
            assert trusting.check_read(leaves[k], proof, tier, roots)
            assert not Verifier(h.global_root).check_read(b"\x00" * 32, proof, tier, roots)
        assert trusting.global_root == h.global_root
        if step % 100 == 0:
            r = h.end_of_period()
            if r["changed"]:
                trusting.trust_layout(h.global_root)
                rebuilding.rebuild_layout(h)
            assert rebuilding.global_root == h.global_root
    sizes = h.tier_sizes()
    assert h.stats["promoted"] > 0 and h.stats["rebuilds"] > 0
    assert sizes["base"] + sizes["overflow"] <= 60 and sizes["cold"] + sizes["base"] + sizes["overflow"] == 600
    hot = [k for k in ranked if h.tier_of(k) != "cold"]
    assert ranked[0] in hot                                      # the most popular key is hot
    assert rebuilding.hashes > trusting.hashes                   # rebuilding the tiers costs


def test_hmt_hot_tier_shortens_proofs_of_popular_keys():
    items = keyed(2000)
    h = HMT(items, theta=0.5, hot_capacity=200, candidates=400, rebuild_every=1)
    popular = [k for k, _ in items[:50]]
    for _ in range(40):
        for k in popular:
            h.read(k)
        h.end_of_period()
    assert all(h.tier_of(k) == "base" for k in popular)
    cold_key = items[-1][0]
    assert h.proof_length(popular[0]) < h.proof_length(cold_key)
