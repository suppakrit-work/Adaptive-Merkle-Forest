"""A provider as a network service: one process, one Storage Service, one DHT node.

    python -m amf.provider.server --name A --port 8001 --data data/A \
                         --peer B=http://localhost:8002 --peer C=http://localhost:8003

POST /provider/<method>, /dht/<method> and /catalog/get take {"args": [...]}
in the wire format of amf.common.wire and answer {"result": ...} or
{"error": type, "message": ...}.  Only the protocol's own methods can be
called; fault injection (corrupt) only when the service runs with --debug.
GET /health and GET /stats answer in plain JSON.

The Migrate Pipe: a migration order goes to the destination, which pulls the
tree from the source named in the order; --peer tells it where each source
listens.  The source still hands the tree over only on the owner's signed
order, which it checks itself.

Calls are served one at a time per provider (one lock), so a provider's state
is never seen half-changed.  Objects live on disk (amf.provider.store); trees, the
Shadow Area and the DHT node live in memory, and HTTP has neither TLS nor
authentication: both are limits of the prototype, not of the design.
"""
import argparse
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from amf.common import wire
from amf.provider import Provider
from amf.provider.client import RemoteProvider
from amf.provider.store import FileStore

PROVIDER_CALLS = {"register_owner", "receive_batch", "receive_row", "store_catalog_entries", "discard_pending",
                  "acknowledged", "write", "delete", "decompose", "compose", "export_tree", "import_from",
                  "pending_challenge", "release", "read", "report_reads", "challenge", "recomputed_leaves",
                  "recomputed_root", "stored_leaves", "stored_root", "halves", "report_mismatches", "stats"}
DEBUG_CALLS = {"corrupt"}
DHT_CALLS = {"get", "digest", "lookup", "row_with_path", "live_rows", "keys"}
MAX_BODY = 256 * 1024 * 1024


class NotAllowed(Exception):
    pass


class Service:
    def __init__(self, provider: Provider, peers: dict, debug: bool = False):
        self.provider = provider
        self.peers = peers                       # name -> RemoteProvider, for the Migrate Pipe
        self.debug = debug
        self.lock = threading.Lock()

    def call(self, part: str, method: str, args: list):
        p = self.provider
        with self.lock:
            if part == "provider":
                if method not in PROVIDER_CALLS and not (self.debug and method in DEBUG_CALLS):
                    raise NotAllowed(method)
                if method == "import_from":
                    order, source_name = args
                    if source_name not in self.peers:
                        raise KeyError(f"no address for provider {source_name}")
                    return p.import_from(order, self.peers[source_name])
                return getattr(p, method)(*args)
            if part == "dht":
                if method not in DHT_CALLS:
                    raise NotAllowed(method)
                if method == "keys":
                    return list(p.dht.rows)
                return getattr(p.dht, method)(*args)
            if part == "catalog" and method == "get":
                (own,) = args
                return p.catalog_copy.get(own, {})
            raise NotAllowed(f"{part}/{method}")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                # keep connections alive between calls
    service: Service = None
    quiet = True

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # no Nagle wait per reply

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.service.provider
        if self.path == "/health":
            self._send(200, json.dumps({"ok": True, "name": p.name, "debug": self.service.debug}).encode())
        elif self.path == "/stats":
            with self.service.lock:
                stats = p.stats()
            self._send(200, json.dumps({"name": p.name, **stats}).encode())
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        parts = self.path.strip("/").split("/")
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, b'{"error": "body too large"}')
            return
        body = self.rfile.read(length)
        if len(parts) != 2:
            self._send(404, b'{"error": "not found"}')
            return
        try:
            args = wire.loads(body).get("args", [])
            reply = {"result": self.service.call(parts[0], parts[1], args)}
        except NotAllowed as e:
            self._send(404, json.dumps({"error": f"not allowed: {e}"}).encode())
            return
        except Exception as e:                   # the caller decides what a failure means
            reply = {"error": type(e).__name__, "message": str(e)}
        self._send(200, wire.dumps(reply))

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)


def make_server(name: str, host: str, port: int, data: str | None, peers: dict, debug: bool = False,
                sync: bool = True):
    objects = FileStore(data, sync=sync) if data else None
    service = Service(Provider(name, objects=objects), peers, debug)
    handler = type("BoundHandler", (Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, service


def main():
    ap = argparse.ArgumentParser(description="Run one AMF provider as an HTTP service.")
    ap.add_argument("--name", required=True, help="the provider's name, as the owner's rows write it")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=None, help="folder for the object store (default: memory only)")
    ap.add_argument("--peer", action="append", default=[], metavar="NAME=URL",
                    help="where another provider listens, for the Migrate Pipe (repeatable)")
    ap.add_argument("--debug", action="store_true", help="allow fault injection (corrupt)")
    ap.add_argument("--no-sync", action="store_true", help="do not fsync each object write")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    peers = {}
    for item in args.peer:
        n, _, url = item.partition("=")
        peers[n] = RemoteProvider(n, url)
    server, _ = make_server(args.name, args.host, args.port, args.data, peers, args.debug, not args.no_sync)
    server.RequestHandlerClass.quiet = not args.verbose
    print(f"provider {args.name} listening on {args.host}:{args.port}"
          f"{' (debug: fault injection on)' if args.debug else ''}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()