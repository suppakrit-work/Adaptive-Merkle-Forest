# Adaptive Merkle Forest (AMF) prototype

Prototype for the CSS 453 paper *Adaptive Merkle Forest: Dynamic Tree Composition
and Decomposition for Scalable Integrity Verification in Multi-Cloud Storage*.
Python 3.11 or newer (tested on 3.11, 3.12 and 3.13); the only library it needs is
`cryptography`. `bench.py`, `workload.py`, `faults.py` and `net_demo.py` derive the owner's
master key from their seed, so a rerun creates the same objects and accesses. Tree identifiers are
random, so owner hashes per access can move by up to about 0.2 between runs (a path in D can
differ by a hash), while read paths and splits stay the same; the times change from machine to machine.

## Layout

The code under `amf/` has one folder per party of the system model (Fig. 1).

| Folder | Files | In the paper |
|---|---|---|
| `amf/common/` | `crypto` identifiers, encryption, leaf hash; `merkle` canonical-shape tree; `bptree` the B+ tree with counts; `index` index rows, DHT node (B+ tree on lo), forest digest D; `orders` signed orders; `wire` the JSON format used on the network | Phase 2, III-B, Definitions 1–2, Lemma 1, Table II |
| `amf/owner/` | `owner` Key/Signing and Verification Manager (Algorithm 2); `catalog` the owner's books (B+ tree on id); `policy` Policy Manager (Algorithm 3); `localize` fault localization (Algorithm 5) | III-A 1, Phases 2–4 and 6 |
| `amf/provider/` | `service` Storage Service (Proof Engine, Shadow Area, Migrate Pipe, DHT node); `store` objects on disk; `server` the service over HTTP; `client` a provider seen over HTTP | III-A 3 |
| `amf/auditor/` | `auditor` the auditor and the data user (Algorithms 1 and 4) | III-A 2, Phases 3 and 5 |
| `baselines/` | the baselines of Section V, one module per paper, built from the same primitives: `wang2011` one Merkle tree per owner (path proofs, the path-update check, fetch-all); `zhu2012` localization by re-challenging halves (CPDP); `mizrahi2024` a reshaping without a check, rebuilt at the verifier (traffic-aware trees, HMT); `shangguan2026` HMT itself, the Huffman-Merkle tree with its cold and hot tiers, Count-Min Sketch, promotion cache and Ratio-Based policy, run on the same access streams; `amf_static` AMF with the policy off | Section V |
| `tests/` | one test file per module | |

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

## Scripts

| Script | What it does | Output |
|---|---|---|
| `demo.py` | the six phases on one machine, step by step | screen |
| `bench.py` | cost of each operation for 10^3 to 10^5 objects, beside the baselines of `baselines/`; with `--state`, only what the owner keeps (catalog, node arrays, rows) | `results/bench.csv`, `results/state.csv` |
| `workload.py` | AMF against the forest that never restructures, one tree, and HMT (with a verifier that trusts its layout changes and one that recomputes them), under uniform and Zipf access; with `--sensitivity`, AMF alone with one policy value changed at a time; with `--merge`, 60% or 85% of the objects deleted and then policy rounds | `results/workload.csv`, `results/sensitivity.csv`, `results/merge.csv` |
| `owners.py` | 1 to 10,000 owners sharing three providers: one owner's upload, read and audit, and a DHT lookup, as the number of owners grows | `results/owners.csv` |
| `net_demo.py` | the owner, a data user and an auditor against three provider services over HTTP: batch loading, skewed reads, policy rounds, a fault and its localization (`--single`: the same run on one tree over all objects, the layout of Wang et al.) | `results/net.csv` |
| `faults.py` | one injected fault per guarantee of Section IV, and the check that catches it | `results/faults.csv` |

Three providers in Docker on one machine:

```
docker compose up --build -d
python net_demo.py
docker compose down -v
```

Providers somewhere else (one per cloud, for example), each started with
`--debug` for the length of the run so that the fault-injection step can run:

```
python -m amf.provider.server --name A --port 8000 --data data/A --debug --peer B=http://<ip-b>:8000 --peer C=http://<ip-c>:8000
python net_demo.py --provider A=http://<ip-a>:8000 --provider B=http://<ip-b>:8000 --provider C=http://<ip-c>:8000 \
                   --files 10000 --batch 200 --reads 2000 --locality 0.1 --rounds 3
```

Objects are loaded as batch writes (`--batch` objects per tree and call, each
checked by the owner as one write); `--batch 0` writes them one at a time.
`--zipf a` draws reads from a Zipf law over the objects, `--locality F` sends
nine reads in ten into the lowest fraction F of each type's identifier range,
and `--rounds` policy rounds follow. `--sequential` sends each new index row
to the DHT nodes one at a time instead of all at once, for comparison.
Three runs make the end-to-end comparison of Section V, each against freshly
started providers: AMF as above; the same with `--rounds 0 --upper 1000000`
(the forest that never restructures); and `--single` (one tree over all
objects at A, no policy). On one machine they are written (`--out`) to
`results/net.csv`, `results/net_static.csv` and `results/net_single.csv`.

The Phase 5 audit challenges `--s` blocks per tree (default 8). For a comparison at equal
coverage (96 blocks, as AMF's 12 trees at s = 8) the static and single runs are repeated with
`--s 32` and `--s 96` (`results/net_static_s32.csv`, `results/net_single_s96.csv`). The
three-cloud runs are in `results/net_cloud*.csv`, described in `results/cloud_setup.md`.

## Limits of the prototype

- A provider keeps its objects on disk but its trees and DHT node in memory, so
  a restarted provider loses its trees. Reset with `docker compose down -v`.
- HTTP has no TLS and no authentication; open port 8000 only to the machines
  that need it.
- `--debug` lets anyone damage an object on purpose (used by `net_demo.py`);
  leave it off outside experiments.
