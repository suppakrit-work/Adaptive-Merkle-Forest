"""The baselines of Section V, one module per scheme, named after the paper it follows.

  wang2011       one Merkle hash tree per owner (Wang et al., TPDS 2011): path
                 proofs, the path-update check, fetch-all as its only way to a
                 bad block
  zhu2012        fault localization by re-challenging halves (Zhu et al., CPDP,
                 TPDS 2012)
  mizrahi2024    a reshaping without a check (Mizrahi et al., ToN 2024; also
                 HMT, Shangguan et al. 2026): the verifier rebuilds the roots
  shangguan2026  HMT, the frequency-aware tiered tree of Shangguan, Yaish and
                 Malkhi (ePrint 2026/1235), run end to end on AMF's workloads
  amf_static     AMF with its Policy Manager switched off (ablation)

Every baseline is built from AMF's own primitives (SHA-256, the Merkle tree
of amf.common.merkle, the providers of amf.provider), so all schemes hash the
same bytes with the same code and differ only in their algorithm.  Each
module's docstring says what it takes from its paper and what it leaves out;
bench.py and workload.py import them and measure them beside AMF, under the
same objects, the same access sequences and on the same machine.
"""
