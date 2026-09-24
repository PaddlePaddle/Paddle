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
"""Partial reader for the ``*.metadata`` file written by ``dist.save_state_dict``.

The file is a plain pickle of :class:`~.metadata.Metadata`, whose three fields
differ enormously in size::

    Metadata(
        state_dict_metadata,  # one entry per tensor key
        storage_metadata,  # one entry per (shard, replica) pair
        flat_mapping,  # one entry per flattened key
    )

``storage_metadata`` is the only structure that binds a shard to a file name, and
it is also by far the largest: on a fully replicated checkpoint it holds
``num_keys * world_size`` entries, which can reach hundreds of MiB and tens of
seconds of unpickling. Callers that only need to know *which tensors exist and
how they are shaped* never touch it.

:func:`load_state_dict_metadata` returns ``Metadata.state_dict_metadata`` alone,
without materializing the rest. It relies on ``state_dict_metadata`` being
pickled immediately before ``storage_metadata``: writing a ``STOP`` opcode over
the opcode that would have pushed the ``"storage_metadata"`` key ends the stream
with exactly the value we want on top of the unpickler's stack, and nothing past
that point is read from disk. Whenever the stream does not match that
expectation the function falls back to a full ``paddle.load``, so calling it is
always safe.
"""

from __future__ import annotations

import os
import pickle
import struct
from typing import TYPE_CHECKING

import paddle
from paddle.distributed.fleet.utils.log_util import logger

from .metadata import LocalTensorMetadata, Metadata

if TYPE_CHECKING:
    from typing import BinaryIO

__all__ = ["load_state_dict_metadata"]

# ``Metadata`` is a plain dataclass, so pickle stores it as a bare instance plus
# a ``__dict__`` state whose keys appear in field declaration order.
_FIELDS: tuple[str, ...] = tuple(Metadata.__dataclass_fields__)
_WANTED_FIELD = "state_dict_metadata"
_STOP_BYTE = pickle.STOP[0]
_PROTO_BYTE = pickle.PROTO[0]
_FRAME_BYTE = pickle.FRAME[0]
_FRAME_HEADER_SIZE = 9  # FRAME opcode + 8-byte little-endian length
_SEARCH_CHUNK = 1 << 20


def _field_after(name: str) -> str | None:
    """The field pickled right after `name`, i.e. where the stream can be cut."""
    if name not in _FIELDS:
        return None
    index = _FIELDS.index(name)
    return _FIELDS[index + 1] if index + 1 < len(_FIELDS) else None


def _key_patterns(name: str) -> list[bytes]:
    """Byte patterns that push the string `name`, across pickle protocols."""
    raw = name.encode("utf-8")
    patterns = []
    if len(raw) < 256:
        patterns.append(pickle.SHORT_BINUNICODE + bytes([len(raw)]) + raw)
    patterns.append(pickle.BINUNICODE + struct.pack("<I", len(raw)) + raw)
    patterns.append(pickle.BINUNICODE8 + struct.pack("<Q", len(raw)) + raw)
    return patterns


def _find_first(buf: bytes, patterns: list[bytes]) -> int:
    hit = -1
    for pattern in patterns:
        found = buf.find(pattern)
        if found != -1 and (hit == -1 or found < hit):
            hit = found
    return hit


def _scan_for_patterns(f: BinaryIO, patterns: list[bytes]) -> int:
    """Offset of the first occurrence of any pattern in `f`, or -1.

    Streams in chunks so the tail of a multi-hundred-MiB file is never read when
    the marker sits near the front, which is the normal case.
    """
    overlap = max(len(pattern) for pattern in patterns) - 1
    f.seek(0)
    tail = b""
    base = 0
    while True:
        chunk = f.read(_SEARCH_CHUNK)
        if not chunk:
            return -1
        buf = tail + chunk
        hit = _find_first(buf, patterns)
        if hit != -1:
            return base + hit
        keep = min(overlap, len(buf))
        tail = buf[len(buf) - keep :]
        base += len(buf) - keep


def _bytes_needed(f: BinaryIO, cut: int) -> int | None:
    """Leading byte count that makes a ``STOP`` planted at `cut` parseable.

    Protocol 4+ wraps the stream in frames whose declared length the unpickler
    insists on reading in full, so the read has to be extended to the end of the
    frame that contains `cut`. Returns ``None`` when the stream is not framed the
    way we expect, or when `cut` does not land inside a frame's payload.
    """
    size = os.fstat(f.fileno()).st_size
    f.seek(0)
    head = f.read(2)
    if len(head) != 2 or head[0] != _PROTO_BYTE:
        return None
    if head[1] < 4:
        # Unframed stream: the cut point is directly addressable.
        return cut + 1

    pos = 2
    while pos < size:
        f.seek(pos)
        header = f.read(_FRAME_HEADER_SIZE)
        if len(header) != _FRAME_HEADER_SIZE or header[0] != _FRAME_BYTE:
            return None
        frame_size = struct.unpack("<Q", header[1:])[0]
        if frame_size == 0:
            return None
        start = pos + _FRAME_HEADER_SIZE
        end = start + frame_size
        if start > cut:
            # `cut` fell in a frame header or an out-of-frame payload.
            return None
        if cut < end:
            return end
        pos = end
    return None


def _as_state_dict_metadata(obj: object) -> dict | None:
    """Recognize and validate the object left on the stack by the early STOP.

    Depending on how pickle batched the ``__dict__`` items, the truncated stream
    yields either the value of ``state_dict_metadata`` directly (``SETITEMS``
    layout, the common case) or a partially filled state dict (``SETITEM``
    layout). Anything else is rejected so the caller can fall back.
    """
    if isinstance(obj, dict) and obj and set(obj).issubset(_FIELDS):
        obj = obj.get(_WANTED_FIELD)
    if not isinstance(obj, dict):
        return None
    for key, value in obj.items():
        if not isinstance(key, str):
            return None
        if isinstance(value, LocalTensorMetadata):
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, LocalTensorMetadata) for item in value
        ):
            continue
        return None
    return obj


def _load_truncated(path: str) -> dict | None:
    """Read only ``state_dict_metadata``, or ``None`` if that is not possible."""
    cut_field = _field_after(_WANTED_FIELD)
    if cut_field is None:
        return None
    try:
        with open(path, "rb") as f:
            cut = _scan_for_patterns(f, _key_patterns(cut_field))
            if cut < 0:
                return None
            needed = _bytes_needed(f, cut)
            if needed is None or needed <= cut:
                return None
            f.seek(0)
            data = bytearray(f.read(needed))
        if len(data) != needed:
            return None
        if _find_first(bytes(data[:cut]), _key_patterns(_WANTED_FIELD)) < 0:
            # ``state_dict_metadata`` is not the field ending at the cut.
            return None
        data[cut] = _STOP_BYTE
        return _as_state_dict_metadata(pickle.loads(bytes(data)))
    except Exception as e:
        logger.debug(
            f"Partial parse of '{path}' failed ({e}), falling back to paddle.load."
        )
        return None


def load_state_dict_metadata(
    path: str, allow_full_load: bool = True
) -> dict[str, list[LocalTensorMetadata]] | None:
    """
    Load only the ``state_dict_metadata`` field of a checkpoint metadata file.

    The returned value is identical to ``paddle.load(path).state_dict_metadata``
    but is obtained without unpickling ``storage_metadata``, which dominates both
    the size and the load time of the file. It maps every tensor key of the
    checkpoint to the list of shard descriptions
    (:class:`~paddle.distributed.flex_checkpoint.dcp.metadata.LocalTensorMetadata`)
    that exist for it globally, so it answers "which tensors are in this
    checkpoint, and how is each one shaped and split" without answering "which
    file holds which shard".

    Args:
        path(str): Path of the ``*.metadata`` file to read.
        allow_full_load(bool): Whether to fall back to a full ``paddle.load``
            when the file cannot be parsed partially. When True (the default)
            the call always returns the metadata. When False the call returns
            None instead of paying the full load, which lets latency-sensitive
            callers choose a different strategy. Default is True.

    Returns:
        dict[str, list[LocalTensorMetadata]], the ``state_dict_metadata`` of the
        checkpoint, or None if partial parsing failed and `allow_full_load` is
        False.

    Examples:
        .. code-block:: pycon

            >>> # doctest: +SKIP('run in distributed mode.')
            >>> import paddle
            >>> import paddle.distributed as dist
            >>> ckpt_path = "./checkpoint"
            >>> w1 = paddle.arange(32).reshape([4, 8])
            >>> dist.save_state_dict({"w1": w1}, ckpt_path)
            >>> state_dict_metadata = dist.load_state_dict_metadata(f"{ckpt_path}/0.metadata")
            >>> # doctest: -SKIP
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Metadata file '{path}' does not exist.")

    state_dict_metadata = _load_truncated(path)
    if state_dict_metadata is not None:
        logger.info(
            f"[flex_checkpoint][metadata_reader] parsed "
            f"{len(state_dict_metadata)} entries of state_dict_metadata from "
            f"'{path}' partially, storage_metadata was not unpickled."
        )
        return state_dict_metadata

    if not allow_full_load:
        return None

    logger.info(
        f"[flex_checkpoint][metadata_reader] could not parse '{path}' "
        "partially, loading the whole metadata file."
    )
    metadata = paddle.load(path)
    if not hasattr(metadata, "state_dict_metadata"):
        raise ValueError(
            f"'{path}' does not contain a Metadata object, got {type(metadata)}."
        )
    return metadata.state_dict_metadata
