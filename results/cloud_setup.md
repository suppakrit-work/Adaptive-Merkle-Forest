# Cloud setup for the three-provider runs

Date: 13 November 2026. Code: commit `22ea4c6` (the same `git archive` was copied to every VM; SHA-256 of the archive `cc842af6d952eca4eaded609f0c273e1176a0aaa19255bee9034fb40def95c09`, checked on each VM).

## Machines

| provider | cloud | region | instance | CPU (lscpu) | vCPU | RAM | fsync rate* |
|---|---|---|---|---|---|---|---|
| A | AWS | ap-southeast-1 (Singapore) | t3.micro | Intel Xeon Platinum 8175M @ 2.50 GHz | 2 | 1 GiB | 313 /s |
| B | Google Cloud | asia-southeast1-b (Singapore) | e2-micro, pd-standard 10 GB | Intel Xeon @ 2.20 GHz | 2 | 1 GB | 387 /s (321–352 /s over 4,000 writes) |
| C | Microsoft Azure | Malaysia West (Kuala Lumpur) | Standard_B2ats_v2 | AMD EPYC 7763 | 2 | 1 GiB | 207 /s |

\* 1,100-byte write, `fsync`, rename, as the provider's store does for every object; 500 writes.

All three: Ubuntu 24.04.5, Python 3.12.3, `cryptography` 50.0.2, providers run as plain processes (`python -m amf.provider.server --debug`, fsync on), not in containers. Azure is in Kuala Lumpur because the Azure for Students subscription does not allow Southeast Asia (Singapore); the deployment therefore spans three providers in two cities. Azure's public-IP idle timeout was raised from 4 to 30 minutes.

The owner (`net_demo.py`) ran on a Windows laptop in Thailand behind a home NAT (Python 3.11). The laptop-only results (`net.csv`, `net_static.csv`, `net_single.csv`) used Docker containers (`python:3.11-slim`) on that laptop.

## Round-trip times (TCP connect, median of 5)

| from \ to | A | B | C |
|---|---|---|---|
| owner (laptop) | 38.4 ms | 37.5 ms | 31.5 ms |
| A | – | 1.7 ms | 6.9 ms |
| B | 2.2 ms | – | 8.2 ms |
| C | 6.9 ms | 8.0 ms | – |

Kept-alive connections stayed usable after 5 minutes idle on all three paths.

## Runs (seed 1, 10,000 objects, batch 200, 2,000 reads, hot range 0.1)

| file | mode | calls equal to the laptop run | localize | fetch-all | migrate A→B | repair |
|---|---|---|---|---|---|---|
| `net_cloud.csv` | AMF, 3 policy rounds | 69/69 steps (5,342 calls) | 9 rounds, 2,620 B | 46.7 KB | 2.6 s | 3.2 s |
| `net_cloud_static.csv` | static forest (`--rounds 0 --upper 1000000`) | 66/66 (3,328) | 11 rounds, 2,940 B | 180.4 KB | 10.0 s | 11.8 s |
| `net_cloud_single.csv` | one tree at A (`--single`) | 61/61 (2,469) | 14 rounds, 3,420 B | 541.1 KB | – | – |
| `net_cloud_smoke.csv` | AMF, 300 objects | – | 6 rounds, 2,140 B | – | – | – |

Providers were reset (process stopped, data deleted) before every run. Every run ended with the rebuilt D matching. Repeatable steps are medians of 20 runs (`runs` column); migrate, repair and the localization after the injected fault run once, and their times follow the destination disk's fsync rate (3,341 objects at ≈ 335 /s ≈ 10 s on B). Peak provider memory in the single run (VmHWM): 50–62 MB.
