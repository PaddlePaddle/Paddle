import paddle.v2.fluid.proto.framework_pb2 as framework_pb2
import paddle.v2.fluid.core as core

import numpy as np

__all__ = [
    'Dtype'
    "as_dtype", "as_pybind_dtype", "bool", "int16", "int32", "int64", "float16",
    "float32", "float64"
]


class Dtype(object):
    """
    Dtype corresponding to Datatype in framework.proto
    """

    def __init__(self, type_index):
        if type_index not in framework_pb2.DataType.values():
            raise TypeError("Unsupported DataType " + str(type_index))
        self.type_index = int(type_index)

    @property
    def is_floating(self):
        floating = set([
            framework_pb2.FP16,
            framework_pb2.FP32,
            framework_pb2.FP64,
        ])
        return self.type_index in floating

    @property
    def is_double(self):
        return self.type_index == framework_pb2.FP64


bool = Dtype(framework_pb2.BOOL)
int16 = Dtype(framework_pb2.INT16)
int32 = Dtype(framework_pb2.INT32)
int64 = Dtype(framework_pb2.INT64)
float16 = Dtype(framework_pb2.FP16)
float32 = Dtype(framework_pb2.FP32)
float64 = Dtype(framework_pb2.FP64)
int = Dtype(framework_pb2.INT32)
float = Dtype(framework_pb2.FP32)

_NP_DTYPE_ = {
    np.bool: bool,
    np.int16: int16,
    np.int32: int32,
    np.int64: int64,
    np.float16: float16,
    np.float32: float32,
    np.float64: float64
}

_STRING_DTYPE_ = {
    "bool": bool,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "float16": float16,
    "float32": float32,
    "float64": float64
}


def as_dtype(dtype):
    if isinstance(dtype, Dtype):
        return dtype
    try:
        return _NP_DTYPE_[dtype]
    except KeyError:
        pass
    try:
        return _STRING_DTYPE_[dtype]
    except KeyError:
        pass
    raise ValueError("Unsupported DataType " + str(dtype))


# TODO(dzhwinter) : should be removed and replace the pybind DataType
_PYBIND_DTYPE_ = {
    bool: core.DataType.BOOL,
    int16: core.DataType.INT16,
    int32: core.DataType.INT32,
    int64: core.DataType.INT64,
    float16: core.DataType.float16,
    float32: core.DataType.float32,
    float64: core.DataType.float64,
}


def as_pybind_dtype(dtype):
    return _PYBIND_DTYPE_[dtype]
