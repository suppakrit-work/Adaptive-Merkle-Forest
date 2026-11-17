# Adaptive Merkle Forest (AMF) Prototype

This repository contains the prototype for our CSS 453 paper *Adaptive Merkle Forest: Dynamic Tree
Composition and Decomposition for Scalable Integrity Verification in Multi-Cloud Storage*. It includes
the AMF implementation, the baseline schemes we compare against, the test suite, and the scripts that
produced every result in the paper.

A step-by-step installation guide is available in [SETUP_GUIDE.pdf](SETUP_GUIDE.pdf).

## Requirements

- Python 3.11 or newer (tested on 3.11, 3.12 and 3.13)
- `cryptography` and `pytest`, installed from `requirements.txt`
- Docker (optional, only for running three providers on one machine)

## Installation and Testing

On Windows:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. All 352 tests
should pass.

To see the protocol run from start to finish, run the demo. It goes through the six phases on one
machine and prints each step:

```
python demo.py
```

## Repository Structure

The code in `amf/` has one folder for each party in the system model (Fig. 1 of the paper).

| Folder | Contents | In the paper |
|---|---|---|
| `amf/common/` | Identifiers, encryption and leaf hashes (`crypto`), the Merkle tree (`merkle`), the B+ tree with counts (`bptree`), index rows, the DHT node and the forest digest D (`index`), signed orders (`orders`), and the message format used on the network (`wire`) | Phase 2, Sec. III-B, Definitions 1–2, Lemma 1, Table II |
| `amf/owner/` | Key/Signing and Verification Manager (`owner`, Algorithm 2), the owner's catalog (`catalog`), Policy Manager (`policy`, Algorithm 3) and fault localization (`localize`, Algorithm 5) | Sec. III-A, Phases 2–4 and 6 |
| `amf/provider/` | Storage Service with the Proof Engine, Shadow Area, Migrate Pipe and DHT node (`service`), object storage on disk (`store`), the HTTP server (`server`) and client (`client`) | Sec. III-A |
| `amf/auditor/` | The auditor and the data user (`auditor`, Algorithms 1 and 4) | Sec. III-A, Phases 3 and 5 |
| `baselines/` | The schemes compared in Section V (see below) | Sec. V |
| `tests/` | One test file for each module | |
| `results/` | Output files of the experiments | Sec. V |

The baselines are our own implementations, built from the same primitives as AMF and run on the same
access streams:

| Module | Scheme | What is implemented |
|---|---|---|
| `wang2011` | Wang et al. | One Merkle tree per owner, path proofs, the path-update check, and fetching all leaf hashes |
| `zhu2012` | CPDP | Fault localization by re-challenging halves |
| `mizrahi2024` | Mizrahi et al. | Traffic-aware restructuring without a check, rebuilt at the verifier |
| `shangguan2026` | HMT | The Huffman-Merkle tree with cold and hot tiers, Count-Min Sketch, promotion cache and the Ratio-Based policy |
| `amf_static` | AMF without the policy | The same forest, but trees are never split or merged |

## Reproducing the Results

Each script below writes one or more CSV files to `results/`. Running a script again replaces its file.

| Command | What it measures | Output | Used in the paper |
|---|---|---|---|
| `python workload.py` | AMF against the static forest, one tree and HMT (with a verifier that trusts HMT's layout changes and one that recomputes them), under uniform and Zipf access | `results/workload.csv` | Figs. 5–6, Table IV |
| `python workload.py --sensitivity` | AMF alone, with one policy value changed at a time | `results/sensitivity.csv` | Table VI |
| `python workload.py --merge` | 60% or 85% of the objects deleted, followed by policy rounds | `results/merge.csv` | Sec. V-A |
| `python bench.py` | Cost of each operation for 10^3 to 10^5 objects, next to the baselines | `results/bench.csv` | Figs. 7–10 |
| `python bench.py --state` | Only what the owner keeps (catalog, node arrays, rows) | `results/state.csv` | Sec. V-A |
| `python owners.py` | 1 to 10,000 owners sharing three providers: upload, read, audit and DHT lookup | `results/owners.csv` | Fig. 11 |
| `python faults.py` | One injected fault for each guarantee in Section IV, and the check that catches it | `results/faults.csv` | Sec. V-A |
| `python net_demo.py` | The owner, a data user and an auditor against three providers over HTTP | `results/net.csv` | Sec. V-B |

`bench.py`, `workload.py`, `faults.py` and `net_demo.py` derive the owner's master key from a fixed seed,
so a second run creates the same objects and accesses. Tree identifiers are still random, so the owner
hashes per access can differ by about 0.2 between runs (a path in D can differ by one hash). Read paths
and the number of splits stay the same. Running times depend on the machine.

## Running Over HTTP

`net_demo.py` runs the owner, a data user and an auditor against three provider services over HTTP. It
loads objects in batches, performs skewed reads, runs policy rounds, injects a fault and localizes it.

### Three providers on one machine (Docker)

```
docker compose up --build -d
python net_demo.py
docker compose down -v
```

This starts providers A, B and C on `127.0.0.1`, ports 8001 to 8003. Without options, `net_demo.py`
runs a short test with 300 objects.

### End-to-end comparison (Section V)

The paper compares three runs. Each run uses `--files 10000 --batch 200 --reads 2000 --locality 0.1`
and starts with fresh providers (`docker compose down -v`, then `docker compose up -d`).

| Run | Extra options | Output |
|---|---|---|
| AMF | `--rounds 3` | `results/net.csv` |
| Forest that never restructures | `--rounds 0 --upper 1000000` | `results/net_static.csv` |
| One tree over all objects at A (layout of Wang et al.) | `--single` | `results/net_single.csv` |

The Phase 5 audit challenges `--s` blocks per tree (default 8). With 12 trees, AMF challenges 96 blocks
in total, so the static and single runs were repeated with `--s 32` and `--s 96` for a comparison at
equal coverage (`results/net_static_s32.csv` and `results/net_single_s96.csv`).

### Providers on separate machines

For the three-cloud runs, each provider ran on its own VM and was started with `--debug` for the length
of the run, so that the fault-injection step could work:

```
python -m amf.provider.server --name A --port 8000 --data data/A --debug --peer B=http://<ip-b>:8000 --peer C=http://<ip-c>:8000
```

The owner then connects to all three:

```
python net_demo.py --provider A=http://<ip-a>:8000 --provider B=http://<ip-b>:8000 --provider C=http://<ip-c>:8000 --files 10000 --batch 200 --reads 2000 --locality 0.1 --rounds 3
```

The results of these runs are in `results/net_cloud*.csv`, and the machines are described in
`results/cloud_setup.md`.

### Main options of `net_demo.py`

| Option | Meaning |
|---|---|
| `--batch N` | Objects per tree in each batch write, checked by the owner as one write (`0` writes one object at a time) |
| `--zipf a` | Draws reads from a Zipf distribution with skew `a` |
| `--locality F` | Sends 9 of every 10 reads to the lowest fraction `F` of each type's identifier range |
| `--rounds N` | Number of policy rounds after the reads |
| `--sequential` | Sends each new index row to the DHT nodes one at a time instead of all at once |
| `--single` | Uses one tree over all objects, without the policy |
| `--s N` | Blocks challenged per tree in the audit |

## Limitations

- A provider keeps its objects on disk but its trees and DHT node in memory, so a restarted provider
  loses its trees. Reset the providers with `docker compose down -v`.
- HTTP is used without TLS or authentication. Open port 8000 only to the machines that need it.
- `--debug` lets anyone damage an object on purpose (`net_demo.py` uses this). Keep it off outside
  experiments.
