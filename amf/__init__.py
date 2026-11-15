"""Adaptive Merkle Forest -- the prototype of the paper, one folder per party of Fig. 1.

    common/    what every party uses: crypto (Phase 2), merkle (Definition 1,
               Lemma 1), bptree (the B+ tree of III-B), index (Table II,
               Definition 2, the DHT node, the forest digest D), orders
               (signed orders of Phase 4), wire (the network format)
    owner/     the Data Owner's three managers, all on the owner's machine:
               owner (Key/Signing and Verification Manager, Algorithm 2),
               catalog (the Verification Manager's books), policy (Policy
               Manager, Phase 4, Algorithm 3), localize (Phase 6, Algorithm 5)
    provider/  every provider's Storage Service: service (Proof Engine, Shadow
               Area, Migrate Pipe, DHT node), store (objects on disk), server
               (the service over HTTP), client (a provider seen over HTTP)
    auditor/   the auditor and the data user (Phase 3)

Each folder imports only from common/ and from itself: the owner, the auditor
and a data user reach a provider only through the calls it answers, whether
it runs in the same process or behind provider/server.py.
"""
