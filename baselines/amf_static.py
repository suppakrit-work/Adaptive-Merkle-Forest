"""AMF with its Policy Manager switched off: the ablation that shows what adapting buys.

The same owner, providers, forest and size bounds as AMF, but no policy round
ever runs: trees are never split, merged or moved, and a write goes into
whichever tree covers its identifier, however large that tree has grown.
Every check of Section IV still applies (the forest is still signed row by
row and under D); what is missing is Phase 4.  workload.py drives this
configuration and AMF with the same access sequence; the difference is the
policy's doing.

One tree, Algorithm 3: the extreme of the same ablation, a forest of a single
tree over all N objects, where every write is checked by recomputing the root
from the catalog (N - 1 hashes at the owner, N leaves rebuilt at the provider)
-- what AMF's own check costs without the forest.  Its costs follow from N
alone, so they are computed rather than run.
"""
from amf.common.merkle import depth_of


def no_policy(owner):
    """Detach any Policy Manager from the owner; returns None in the place of one."""
    owner.policy = None
    return None


def one_tree_costs(n: int, reads: int, writes: int) -> dict:
    """Per-workload totals for one tree of n objects checked with Algorithm 3: n - 1 hashes
    at the owner and n leaves at the provider per write, a path of log2 n per read."""
    d = depth_of(n)
    return dict(owner_hashes=writes * (n - 1), provider_leaves=writes * n, read_path=reads * d,
                reads=reads, writes=writes, owner_ms=None, provider_ms=None, splits=0, merges=0,
                trees=1, mean_tree=n)
