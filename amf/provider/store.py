"""The provider's object store on disk: one file per object, named by its identifier.

A write goes to a temporary file first and is renamed over the old one, so a
crash leaves either the old ciphertext or the new one, never half of each.
The store behaves like a dict from identifier to ciphertext, which is all
amf.provider.Provider asks of it.  Only the objects live here; the trees,
the Shadow Area and the DHT node stay in memory (a known limit of the
prototype: a provider that restarts keeps its objects but not its trees).
"""
import os
from collections.abc import MutableMapping

from amf.common.crypto import ID_LEN


class FileStore(MutableMapping):
    def __init__(self, root: str, sync: bool = True):
        self.root = root
        self.sync = sync                     # fsync each write before the rename
        os.makedirs(root, exist_ok=True)

    def _path(self, obj_id) -> str:
        if not isinstance(obj_id, bytes) or len(obj_id) != ID_LEN:
            raise KeyError(obj_id)
        return os.path.join(self.root, obj_id.hex())

    def __getitem__(self, obj_id) -> bytes:
        try:
            with open(self._path(obj_id), "rb") as f:
                return f.read()
        except FileNotFoundError:
            raise KeyError(obj_id) from None

    def __setitem__(self, obj_id, ct: bytes) -> None:
        path = self._path(obj_id)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(ct)
            if self.sync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)

    def __delitem__(self, obj_id) -> None:
        try:
            os.remove(self._path(obj_id))
        except FileNotFoundError:
            raise KeyError(obj_id) from None

    def __contains__(self, obj_id) -> bool:
        try:
            return os.path.exists(self._path(obj_id))
        except KeyError:
            return False

    def __iter__(self):
        for name in os.listdir(self.root):
            if len(name) == 2 * ID_LEN:
                yield bytes.fromhex(name)

    def __len__(self) -> int:
        return sum(1 for _ in self)