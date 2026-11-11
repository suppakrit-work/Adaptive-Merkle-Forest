"""Fault localization by re-challenging halves, as in CPDP (Zhu et al., TPDS 2012).

Y. Zhu, H. Hu, G.-J. Ahn and M. Yu, "Cooperative Provable Data Possession for
Integrity Verification in Multicloud Storage", IEEE Trans. Parallel Distrib.
Syst. 23(12), 2012, verify a file spread over several clouds with homomorphic
tags and one aggregated response per challenge.  A response that fails says
nothing about which block is bad, and the aggregation hides the block's
location on purpose.  Section 5.2 of the paper proposes to find it anyway: take
a part of the challenged indexes as a new challenge, halve it each time, until
the bad block is found, in O(log n) challenges.

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
    and the search continues in whichever half is bad.  Returns (position, re-challenges, blocks)."""
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
