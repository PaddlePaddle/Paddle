import os, re
import threading
from contextlib import contextmanager

_local = threading.local()


class Nameprefix(object):
    global _local

    def __init__(self, prefix, reset=False):
        if not hasattr(_local, "nameprefix") or reset:
            _local.nameprefix = []
        # assert(isinstance(prefix, basestring), "prefix must be string")
        if prefix not in _local.nameprefix:
            _local.nameprefix.append(prefix)
        self.nameprefix = _local.nameprefix

    def __enter__(self):
        return self.nameprefix

    def __exit__(self, exc_type, exc_value, traceback):
        self.nameprefix.pop(len(self.nameprefix) - 1)


nameprefix = Nameprefix


def current_nameprefix():
    global _local
    return "/".join(_local.nameprefix) + "/"
