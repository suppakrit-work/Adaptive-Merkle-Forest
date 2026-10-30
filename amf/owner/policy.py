"""The Policy Manager -- Phase 4, the rules of split and merge.

Let R_L and R_R be the accesses, reads and writes, a tree has seen on its first
ceil(n/2) leaves and on the rest, R_a and R_b those of two adjacent trees, and

    hot(x, y)  :=  x + y >= mu  and  max(x, y) > gamma * min(x, y)

A tree is split, and two adjacent trees at the same provider are merged, when

    n > N_up  or  (n >= 2 N_low  and  hot(R_L, R_R))          split
    n_a + n_b < N_low  and not  hot(R_a, R_b)                  merge

A split is made at the median, where the check is one hash.  The size test is
also made on the count a write would produce, before the write, so no tree is
ever taken past N_up.  A migration is not driven by the counts (owner.evacuate).
A hysteresis rule keeps N_low well under half of N_up and forbids touching the
same tree twice within Delta_min policy rounds.

Reads come from the providers' reports, per half of each tree, the owner's
included; writes are the ones the owner made (owner.write_counts).
Parameters (Table II): N_up 1000, N_low 250, gamma 4, mu 50, Delta_min 3.
"""
from dataclasses import dataclass, field


@dataclass
class Decision:
    kind: str            # "split" | "merge"
    trees: tuple         # tree ids involved
    reason: str


def hot(x: int, y: int, gamma: float, mu: int) -> bool:
    """Two halves used enough for the counts to mean something, and one much more than the other."""
    return x + y >= mu and max(x, y) > gamma * min(x, y)


@dataclass
class PolicyManager:
    owner: object
    upper: int = 1000          # N_up
    lower: int = 250           # N_low
    gamma: float = 4.0
    mu: int = 50
    interval: int = 3          # Delta_min, in policy rounds
    round_no: int = 0
    reads: dict = field(default_factory=dict)            # tree id -> [left, right], provider-reported
    last_touched: dict = field(default_factory=dict)     # tree id -> policy round

    def __post_init__(self):
        if self.lower * 4 > self.upper:
            raise ValueError("N_low must stay well under half of N_up")
        self.owner.policy = self

    # -- the counts the rules read -------------------------------------------------

    def collect_reads(self) -> None:
        """Fold in what each provider reports for the owner's trees."""
        for prov in self.owner.providers.values():
            for (own, tid), (l, r) in prov.report_reads().items():
                if own != self.owner.own:
                    continue
                halves = self.reads.setdefault(tid, [0, 0])
                halves[0] += l
                halves[1] += r

    def accesses(self, tree_id: bytes) -> tuple[int, int]:
        """(R_L, R_R): reads reported by the providers plus writes the owner made."""
        rd = self.reads.get(tree_id, [0, 0])
        wr = self.owner.write_counts.get(tree_id, [0, 0])
        return rd[0] + wr[0], rd[1] + wr[1]

    def touchable(self, *tree_ids) -> bool:
        return all(self.round_no - self.last_touched.get(t, -10**9) >= self.interval for t in tree_ids)

    # -- the rules ------------------------------------------------------------------

    def decide(self) -> list[Decision]:
        rows = sorted(self.owner.live_rows(), key=lambda r: r.lo)
        out, taken = [], set()
        for r in rows:
            if not self.touchable(r.tree_id):
                continue
            rl, rr = self.accesses(r.tree_id)
            if r.count > self.upper:
                out.append(Decision("split", (r.tree_id,), f"n = {r.count} > N_up"))
                taken.add(r.tree_id)
            elif r.count >= 2 * self.lower and hot(rl, rr, self.gamma, self.mu):
                out.append(Decision("split", (r.tree_id,), f"hot halves {rl}/{rr}"))
                taken.add(r.tree_id)
        for a, b in zip(rows, rows[1:]):
            if a.tree_id in taken or b.tree_id in taken:
                continue
            if a.hi != b.lo or a.provider != b.provider or a.count + b.count >= self.lower:
                continue
            ra, rb = sum(self.accesses(a.tree_id)), sum(self.accesses(b.tree_id))
            if hot(ra, rb, self.gamma, self.mu) or not self.touchable(a.tree_id, b.tree_id):
                continue
            out.append(Decision("merge", (a.tree_id, b.tree_id), f"n_a + n_b = {a.count + b.count} < N_low"))
            taken.update((a.tree_id, b.tree_id))
        return out

    def before_write(self, row, obj_id: bytes):
        """The size test on the count a write would produce: split first if it would pass N_up.
        Returns the row that covers obj_id afterwards."""
        if row.count + 1 > self.upper and row.count >= 2:
            a, b, _ = self.owner.split(row.tree_id)
            self._after_split(row.tree_id, a, b)
            return self.owner.row_for(obj_id)
        return row

    # -- one policy round ----------------------------------------------------------

    def tick(self) -> list[Decision]:
        self.round_no += 1
        self.collect_reads()
        done = []
        for d in self.decide():
            if d.kind == "split":
                a, b, _ = self.owner.split(d.trees[0])
                self._after_split(d.trees[0], a, b)
            else:
                new = self.owner.merge(*d.trees)
                self._touch(new.tree_id, *d.trees)
                for t in d.trees:
                    self.reads.pop(t, None)
            done.append(d)
        return done

    def _after_split(self, old_id, a, b) -> None:
        self._touch(old_id, a.tree_id, b.tree_id)
        self.reads.pop(old_id, None)                     # the halves start counting afresh

    def _touch(self, *tree_ids) -> None:
        for t in tree_ids:
            self.last_touched[t] = self.round_no