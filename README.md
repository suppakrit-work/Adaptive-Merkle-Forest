# Adaptive Merkle Forest (AMF) prototype

Prototype for the CSS 453 paper *Adaptive Merkle Forest: Dynamic Tree Composition
and Decomposition for Scalable Integrity Verification in Multi-Cloud Storage*.
Python 3.11; the only library it needs is `cryptography`.

## Layout

The code under `amf/` has one folder per party of the system model (Fig. 1).

| Folder | Files | In the paper |
|---|---|---|
| `amf/common/` | `crypto` identifiers, encryption, leaf hash; `merkle` canonical-shape tree; `bptree` the B+ tree with counts; `index` index rows, DHT node (B+ tree on lo), forest digest D; `orders` signed orders; `wire` the JSON format used on the network | Phase 2, III-B, Definitions 1–2, Lemma 1, Table I |
| `amf/owner/` | `owner` Key/Signing and Verification Manager (Algorithm 3); `catalog` the owner's books (B+ tree on id); `policy` Policy Manager (Algorithm 2); `localize` fault localization (Algorithm 5) | III-A 1, Phases 2–4 and 6 |
| `amf/provider/` | `service` Storage Service (Proof Engine, Shadow Area, Migrate Pipe, DHT node); `store` objects on disk; `server` the service over HTTP; `client` a provider seen over HTTP | III-A 3 |
| `amf/auditor/` | `auditor` the auditor and the data user (Algorithms 1 and 4) | III-A 2, Phases 3 and 5 |
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
| `demo.py` | the five phases on one machine, step by step | screen |
| `bench.py` | cost of each operation for 10^3 to 10^5 objects, with the baselines of Section V | `results/bench.csv` |
| `workload.py` | AMF against a forest that never restructures, under uniform and Zipf access | `results/workload.csv` |
| `owners.py` | 1 to 10,000 owners sharing three providers: one owner's upload, read and audit, and a DHT lookup, as the number of owners grows | `results/owners.csv` |
| `net_demo.py` | the owner, a data user and an auditor against three provider services over HTTP | `results/net.csv` |

Three providers in Docker on one machine:

```
docker compose up --build -d
python net_demo.py
docker compose down -v
```

Providers somewhere else (one per cloud, for example):

```
python -m amf.provider.server --name A --port 8000 --data data/A --peer B=http://<ip-b>:8000 --peer C=http://<ip-c>:8000
python net_demo.py --provider A=http://<ip-a>:8000 --provider B=http://<ip-b>:8000 --provider C=http://<ip-c>:8000
```

`net_demo.py --sequential` sends each new index row to the DHT nodes one at a
time instead of all at once, for comparison.

## Limits of the prototype

- A provider keeps its objects on disk but its trees and DHT node in memory, so
  a restarted provider loses its trees. Reset with `docker compose down -v`.
- HTTP has no TLS and no authentication; open port 8000 only to the machines
  that need it.
- `--debug` lets anyone damage an object on purpose (used by `net_demo.py`);
  leave it off outside experiments.
