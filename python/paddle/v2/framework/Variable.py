# variable. only for test namespace
# import namespace as space
import threading
_local = threading.local()


def gen_id():
    global _local
    if not hasattr(_local, "uid"):
        _local.uid = 0
    _local.uid += 1
    return _local.uid


scope = dict()


def Variable(name):
    if name not in scope:
        scope[name] = [Parameter(name), 0]
        return scope[name][0]
    else:
        scope[name][1] += 1
        return scope[name][0]


class Parameter(object):
    def __init__(self, name):
        self._name = name
        self._uid = gen_id()

    def __add__(self, other):
        if isinstance(other, Parameter):
            return Parameter(gen_id())

    def __str__(self):
        return self._name
