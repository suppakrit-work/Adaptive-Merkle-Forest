"""A provider seen over the network: the calls of amf.provider.Provider, sent as HTTP POSTs.

The owner, a reader, an auditor or the Localizer cannot tell a RemoteProvider
from a local Provider: the same methods take the same arguments and give the
same answers, encoded with amf.common.wire.  Two attributes are views rather than
data: .dht asks the provider's DHT node, and .catalog_copy asks for the
encrypted catalog copy of one owner.  An error raised at the provider comes
back as the same exception type where it makes sense (PermissionError for an
order that does not verify, ValueError for a tree that does not match, ...);
anything else, a refusal or a dead connection among them, is a RemoteError.

Every call counts the bytes it sent and received, so the cost of an operation
over a real network can be read off afterwards (see net_demo.py).
"""
import http.client
import socket
import threading
import time
import urllib.parse

from amf.common import waits, wire

ERRORS = {e.__name__: e for e in (PermissionError, ValueError, KeyError, IndexError, RuntimeError,
                                  TimeoutError)}


class RemoteError(RuntimeError):
    """The provider refused, failed, or could not be reached."""


class Connection:
    """One kept-alive HTTP connection, used by one call at a time."""

    def __init__(self, url: str, timeout: float = 120.0):
        u = urllib.parse.urlsplit(url)
        if u.scheme != "http" or not u.hostname:
            raise ValueError(f"not an http:// address: {url}")
        self.host, self.port, self.timeout = u.hostname, u.port or 80, timeout
        self.lock = threading.Lock()
        self.conn = None
        self.calls = self.sent = self.received = 0

    def request(self, method: str, path: str, body: bytes = b"") -> bytes:
        with self.lock:
            t0 = time.perf_counter()
            for attempt in (1, 2):
                try:
                    if self.conn is None:
                        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
                        self.conn.connect()
                        self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.conn.request(method, path, body=body, headers={"Content-Type": "application/json"})
                    resp = self.conn.getresponse()
                    data = resp.read()
                    break
                except (http.client.RemoteDisconnected, ConnectionResetError, BrokenPipeError):
                    self.close()                    # a kept-alive connection the server had dropped
                    if attempt == 2:
                        raise RemoteError(f"{self.host}:{self.port} closed the connection") from None
                except OSError as e:
                    self.close()
                    raise RemoteError(f"{self.host}:{self.port}: {e}") from None
            self.calls += 1
            self.sent += len(body)
            self.received += len(data)
            waits.note(time.perf_counter() - t0)
            if resp.status != 200:
                raise RemoteError(f"HTTP {resp.status} for {path}: {data[:200]!r}")
            return data

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None


class RemoteProvider:
    def __init__(self, name: str, url: str, timeout: float = 120.0):
        self.name = name
        self.url = url
        self.net = Connection(url, timeout)
        self.dht = RemoteDHT(self)
        self.catalog_copy = RemoteCatalogCopy(self)

    def call(self, part: str, method: str, *args):
        reply = wire.loads(self.net.request("POST", f"/{part}/{method}", wire.dumps({"args": list(args)})))
        if "error" in reply:
            raise ERRORS.get(reply["error"], RemoteError)(reply["message"])
        return reply["result"]

    def health(self) -> dict:
        return wire.loads(self.net.request("GET", "/health"))

    # -- the index copy, catalog copy, pending trees (Phase 1, 2, 4) ------------------
    def register_owner(self, pk):
        return self.call("provider", "register_owner", pk)

    def receive_batch(self, rows, digest=None):
        return self.call("provider", "receive_batch", list(rows), digest)

    def receive_row(self, row):
        return self.call("provider", "receive_row", row)

    def store_catalog_entries(self, own, entries):
        return self.call("provider", "store_catalog_entries", own, entries)

    def discard_pending(self, own, *tree_ids):
        return self.call("provider", "discard_pending", own, *tree_ids)

    def acknowledged(self, own, tree_id, root):
        return self.call("provider", "acknowledged", own, tree_id, root)

    # -- writes and restructuring --------------------------------------------------------
    def write(self, obj_id, ct):
        return self.call("provider", "write", obj_id, ct)

    def write_many(self, items):
        return self.call("provider", "write_many", list(items))

    def delete(self, obj_id):
        return self.call("provider", "delete", obj_id)

    def decompose(self, order):
        return self.call("provider", "decompose", order)

    def compose(self, order):
        return self.call("provider", "compose", order)

    def export_tree(self, order):
        return self.call("provider", "export_tree", order)

    def import_from(self, order, source):
        """The destination pulls the tree from the source itself, through the Migrate Pipe."""
        return self.call("provider", "import_from", order, source.name)

    def pending_challenge(self, own, tree_id, positions):
        return self.call("provider", "pending_challenge", own, tree_id, list(positions))

    def release(self, order):
        return self.call("provider", "release", order)

    # -- the Proof Engine and fault localization -----------------------------------------
    def read(self, obj_id):
        return self.call("provider", "read", obj_id)

    def report_reads(self, own):
        return self.call("provider", "report_reads", own)

    def challenge(self, own, tree_id, positions):
        return self.call("provider", "challenge", own, tree_id, list(positions))

    def recomputed_leaves(self, own, tree_id):
        return self.call("provider", "recomputed_leaves", own, tree_id)

    def recomputed_root(self, own, tree_id):
        return self.call("provider", "recomputed_root", own, tree_id)

    def stored_leaves(self, own, tree_id):
        return self.call("provider", "stored_leaves", own, tree_id)

    def stored_root(self, own, tree_id):
        return self.call("provider", "stored_root", own, tree_id)

    def recount(self, own):
        return self.call("provider", "recount", own)

    def halves(self, own, tree_id, k, lo, hi):
        return self.call("provider", "halves", own, tree_id, k, lo, hi)

    def round_halves(self, own, tree_id, nodes):
        return self.call("provider", "round_halves", own, tree_id, list(nodes))

    def report_mismatches(self, own, tree_id):
        return self.call("provider", "report_mismatches", own, tree_id)

    def stats(self):
        return self.call("provider", "stats")

    def corrupt(self, obj_id, at=0):
        """Fault injection; the provider accepts it only when started with --debug."""
        return self.call("provider", "corrupt", obj_id, at)


class RemoteDHT:
    """The provider's DHT node, asked one question at a time."""

    def __init__(self, prov: RemoteProvider):
        self._p = prov

    def get(self, own, tree_id):
        return self._p.call("dht", "get", own, tree_id)

    def digest(self, own):
        return self._p.call("dht", "digest", own)

    def lookup(self, obj_id):
        return self._p.call("dht", "lookup", obj_id)

    def row_with_path(self, own, tree_id):
        return self._p.call("dht", "row_with_path", own, tree_id)

    def route(self, obj_id):
        return self._p.call("dht", "route", obj_id)

    def live_rows(self, own=None):
        return self._p.call("dht", "live_rows", own)

    def tree_ids(self, own):
        return self._p.call("dht", "tree_ids", own)


class RemoteCatalogCopy:
    """The encrypted catalog copy kept at the provider, one owner at a time."""

    def __init__(self, prov: RemoteProvider):
        self._p = prov

    def get(self, own, default=None):
        copy = self._p.call("catalog", "get", own)
        return copy if copy else default
