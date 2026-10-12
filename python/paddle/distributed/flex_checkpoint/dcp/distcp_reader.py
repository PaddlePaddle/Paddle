# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Header-only reader for the ``*.distcp`` files written by ``dist.save_state_dict``.

A ``.distcp`` file is a plain pickle of a flat dict::

    {"<tensor_key>": numpy.core.numeric._frombuffer(<payload>, dtype, shape, order),
     ...,
     "StructuredToParameterName@@": {...}}

Two properties of the pickle format make the keys and shapes readable without
touching the weights:

1. Every weight payload carries an explicit length in its opcode, so its size is
   known before its bytes are.
2. Pickle writes any payload of at least ``_FRAME_SIZE_TARGET`` (64 KiB) outside
   the frame stream, which means the file object is positioned exactly on those
   bytes and they can be ``seek()``-ed over.

:func:`scan_tensor_shapes` therefore costs a few hundred bytes per tensor instead
of the whole file. It deliberately reports what the file itself claims to hold
rather than what the checkpoint metadata says it holds, so a file that was
truncated or never fully written is detected rather than trusted.
"""

from __future__ import annotations

import pickle
import struct

_BYTEARRAY8 = pickle.BYTEARRAY8[0]
_BINBYTES8 = pickle.BINBYTES8[0]
# Sentinel telling "the unpickler does not expose its framing state".
_UNKNOWN_FRAME = object()


class _Payload:
    """Stands in for a weight payload; only its length is retained."""

    __slots__ = ("nbytes",)

    def __init__(self, nbytes: int):
        self.nbytes = nbytes


class _Global:
    """Stands in for any global the stream references, so nothing is imported."""

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __call__(self, *args):
        return _Call(self.name, args)


class _Call:
    __slots__ = ("name", "args")

    def __init__(self, name: str, args: tuple):
        self.name = name
        self.args = args

    def __setstate__(self, state):
        # ``numpy.dtype`` rebuilds itself through BUILD; its state is irrelevant
        # here, but swallowing it keeps the walk going.
        pass


class _HeaderUnpickler(pickle._Unpickler):
    """Unpickler that skips payload bytes and imports nothing."""

    def __init__(self, f):
        super().__init__(f)
        self._raw = f
        self.dispatch = dict(pickle._Unpickler.dispatch)
        self.dispatch[_BYTEARRAY8] = _HeaderUnpickler._load_payload
        self.dispatch[_BINBYTES8] = _HeaderUnpickler._load_payload

    def _load_payload(self):
        nbytes = struct.unpack("<Q", self.read(8))[0]
        frame = getattr(self._unframer, "current_frame", _UNKNOWN_FRAME)
        if frame is None:
            # Out-of-frame payload: the raw file is positioned right on it.
            self._raw.seek(nbytes, 1)
        else:
            # In-frame payload (below 64 KiB), already buffered; or an unpickler
            # whose internals we cannot inspect. Either way, read and drop it.
            self.read(nbytes)
        self.append(_Payload(nbytes))

    def find_class(self, module, name):
        return _Global(f"{module}.{name}")


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    """The shape of a pickled numpy array, or ``None`` if `value` is not one."""
    if not isinstance(value, _Call) or not value.name.endswith("_frombuffer"):
        return None
    if len(value.args) < 3 or not isinstance(value.args[0], _Payload):
        return None
    shape = value.args[2]
    if not isinstance(shape, (tuple, list)):
        return None
    return tuple(int(dim) for dim in shape)


def scan_tensor_shapes(path: str) -> dict[str, tuple[int, ...]]:
    """
    Return ``{tensor_key: shape}`` for `path` without reading any weight bytes.

    Non-tensor top-level entries, such as ``StructuredToParameterName@@``, are
    skipped. The stream is walked all the way to ``STOP``, so a truncated or
    corrupt file raises instead of being silently accepted.
    """
    with open(path, "rb") as f:
        obj = _HeaderUnpickler(f).load()
    if not isinstance(obj, dict):
        raise ValueError(
            f"'{path}' does not hold a dict at top level, got {type(obj)}."
        )
    shapes = {}
    for key, value in obj.items():
        shape = _tensor_shape(value)
        if shape is not None and isinstance(key, str):
            shapes[key] = shape
    return shapes
