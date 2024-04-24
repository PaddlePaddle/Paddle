from __future__ import annotations

from .framework.dtype import (
    bfloat16 as bfloat16,
    bool as bool,
    complex64 as complex64,
    complex128 as complex128,
    dtype as dtype,
    finfo as finfo,
    float16 as float16,
    float32 as float32,
    float64 as float64,
    iinfo as iinfo,
    int8 as int8,
    int16 as int16,
    int32 as int32,
    int64 as int64,
    uint8 as uint8,
)

class Tensor: ...

def to_tensor(data, dtype=None, place=None, stop_gradient=True) -> Tensor: ...

class CPUPlace: ...

class CUDAPlace:
    def __init__(self, id: int, /) -> None: ...

class CUDAPinnedPlace: ...

class NPUPlace:
    def __init__(self, id: int, /) -> None: ...

class IPUPlace: ...

class CustomPlace:
    def __init__(self, name: str, id: int, /) -> None: ...

class MLUPlace:
    def __init__(self, id: int, /) -> None: ...

class XPUPlace:
    def __init__(self, id: int, /) -> None: ...
