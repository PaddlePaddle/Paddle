import proto.framework_pb2 as framework_pb2
from . import core

import numpy as np

__all__ = [
    "Dtype",
    "as_dtype"
    "int32",
    "float32",
]

# BOOL = 0;
# INT16 = 1;
# INT32 = 2;
# INT64 = 3;
# FP16 = 4;
# FP32 = 5;
# FP64 = 6;


class Dtype(object):
    """
    Dtype corresponding to Datatype in framework.proto
    """

    def __init__(self, type_index):
        if type_index not in framework_pb2.DataType.values():
            raise TypeError("InValid DataType. %s" % (str(type_index)))
        self.type_index = int(type_index)

    def is_floating(self):
        floating = set([
            framework_pb2.FP16,
            framework_pb2.FP32,
            framework_pb2.FP64,
        ])
        return self.type_index in floating

    @property
    def is_double(self):
        return self.type_index == framework_pb2.FP32


int32 = Dtype(framework_pb2.INT32)
float32 = Dtype(framework_pb2.FP32)

_NP_DTYPE_ = {np.int32: int32, np.float32: float32}

_STRING_DTYPE_ = {"int32": int32, "float32": float32}


def as_dtype(dtype):
    if isinstance(Dtype):
        return dtype
    try:
        return _NP_DTYPE_[dtype]
    except KeyError:
        pass
    try:
        return _STRING_DTYPE_[dtype]
    except KeyError:
        pass
    raise ValueError("Not supported dtype " + str(dtype))
