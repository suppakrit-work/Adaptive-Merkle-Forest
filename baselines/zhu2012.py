"""Fault localization by re-challenging halves, as in CPDP (Zhu et al., TPDS 2012).

Y. Zhu, H. Hu, G.-J. Ahn and M. Yu, "Cooperative Provable Data Possession for
Integrity Verification in Multicloud Storage", IEEE Trans. Parallel Distrib.
Syst. 23(12), 2012, verify a file spread over several clouds with homomorphic
tags and one aggregated response per challenge.  A response that fails says
nothing about which block is bad, and the aggregation hides the block's
location on purpose.  Section 5.2 of the paper says, in one paragraph, that the
block can be found anyway: take "the partial set of challenge indexes as a new
challenge set" and repeat "until the bad block is found", in O(log n) steps.
The paper fixes neither the part taken nor the starting set (its challenge is a
random sample of t = n*w blocks); halving, the first half of ceil(size/2)
blocks, and starting from the whole tree of n blocks are our concretization of
that paragraph, the one that gives its O(log n).

This module runs that search on AMF's trees.  A challenge covers a range of
positions; the provider answers for every position in it (AMF's challenge call
stands in for the aggregated response) and the verifier checks each answer.
Counted, as the paper does, are the challenges issued after the one that
failed and the blocks the provider had to touch to answer them -- the two
quantities that grow with n.  The tags, pairings and the organizer of CPDP
are not implemented, so its running times and response sizes are not
reproduced here; the paper's own figures must be cited for those.
"""
from amf.common.merkle import depth_of


def expected(n: int) -> tuple[int, int]:
    """(re-challenges, blocks touched) to find one bad block among n by halving, counting the
    challenge over all n that failed first: at most ceil(log2 n) re-challenges (one fewer when the
    fault sits on a shorter path) and at most n + sum ceil(n / 2^k) blocks; the exact counts depend
    on which halves the fault falls in, so localize() measures them."""
    d = depth_of(n)
    return d, n + sum(-(-n // 2 ** k) for k in range(1, d + 1))


def localize(check_range, n: int) -> tuple[int, int, int]:
    """Find one bad block among n by re-challenging halves.  check_range(lo, hi) must challenge the
    positions [lo, hi) and return True iff every answer verifies.  The challenge over all n blocks
    is the one that failed and is counted; then the first half of the suspect range is challenged
    and the search continues in whichever half is bad.  Returns (position, re-challenges, blocks).
    With several bad blocks it returns the leftmost; it must be called after a failed challenge,
    since on a range with no bad block it returns n - 1 without having checked it."""
    if n < 1:
        raise ValueError("nothing to localize")
    rounds, blocks, lo, hi = 0, n, 0, n
    while hi - lo > 1:
        mid = lo + (hi - lo + 1) // 2                  # the first half, ceil(size / 2) blocks
        rounds += 1
        blocks += mid - lo
        if check_range(lo, mid):
            lo = mid                                   # the first half is fine: the fault is in the other
        else:
            hi = mid
    return lo, rounds, blocks
