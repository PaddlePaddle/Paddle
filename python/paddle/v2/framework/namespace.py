import os, re
import threading
from contextlib import contextmanager
import inspect

_local = threading.local()


def current_namespace():
    global _local
    if not hasattr(_local, "namespace"):
        _local.namespace = ""
    return _local.namespace


# @contextmanager
# def namespace(prefix, reset=False):
#   global _local
#   _old_namespace = current_namespace()
#   if reset:
#     _local.namespace = ""
#   else:
#     _local.namespace += prefix + os.sep
#   yield _local.namespace
# try:
#   yield _local.namespace
# finally:
#   if not _local.namespace.endswith(prefix+os.sep):
#     _local.namespace = _old_namespace
#     raise ValueError("create namespace failed %s", prefix)

# except ValueError as e:
#   print e
# finally :
#   print "namespace invalid"


class Namespace(object):
    def __init__(self, prefix, reset=False):
        _local = threading.local()
        if not hasattr(_local, "namespace"):
            _local.namespace = ""
        self._namespace = _local.namespace

    def __enter__(self):
        print "enter"
        local_namespace = inspect.getframeinfo(inspect.currentframe().f_back)
        i = 0
        for line in local_namespace:
            print str(i) + " ", line
            i += 1
        # print local_namespace
        return self._namespace

    def __exit__(self, type, value, trackback):
        print "exit"


with Namespace("dzh") as net:
    print net
