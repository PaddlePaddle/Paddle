# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""Multi-threaded payload reading for ``paddle.load``.

``paddle.load`` is bound by moving tensor bytes with a single thread, not by
parsing pickle: the metadata is a few KB while the payloads are GBs of raw bytes.
The unpickler asks the file object to ``readinto()`` every payload written
outside a pickle frame, so ``_HolePunchFile`` records those requests and skips
them; the returned numpy arrays are zero-copy views over the still empty
buffers, which ``_fill_holes`` then reads in parallel with ``os.preadv``.
The file format is untouched and the fast path is opt-in.
"""

from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from .restricted_unpickler import safe_load_pickle

if TYPE_CHECKING:
    from collections.abc import Iterable

# Smaller payloads live inside pickle frames and never reach ``readinto``.
# Well above the 64KB framing target: intercepting a frame-local read would
# corrupt the recorded offsets.
_MIN_PAYLOAD_SIZE = 1 << 20

# Bytes already prefetched by the buffered reader are served through ``read``
# and stay on the serial path, so keep the scan buffer small.
_SCAN_BUFFER_SIZE = 8192

# Large payloads are split so that a checkpoint with few tensors still keeps the
# pool busy. 32MB measured best (64MB limits parallelism, 16MB fragments IO).
_READ_CHUNK_SIZE = 32 << 20


def _resolve_num_workers(num_workers: int | None) -> int:
    """Validate ``num_workers``; ``None``/``0``/``1`` all mean serial."""
    if num_workers is None:
        return 0
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError(
            "The 'num_workers' of `paddle.load` should be int, but received "
            f"{type(num_workers).__name__}."
        )
    if num_workers < 0:
        raise ValueError(
            "The 'num_workers' of `paddle.load` should be >= 0, but received "
            f"{num_workers}."
        )
    return num_workers


class _HolePunchFile:
    """Records large ``readinto`` requests instead of serving them."""

    def __init__(self, f, thresh: int = _MIN_PAYLOAD_SIZE) -> None:
        self._f = f
        self._thresh = thresh
        self.holes: list[tuple[int, Any]] = []

    def read(self, n: int = -1) -> bytes:
        return self._f.read(n)

    def readline(self) -> bytes:
        return self._f.readline()

    def peek(self, n: int = 1) -> bytes:
        return self._f.peek(n)

    def readinto(self, b) -> int:
        n = len(b)
        if n >= self._thresh:
            # ``b`` views the buffer the numpy array will use; keeping it here
            # also keeps it alive until the fill pass.
            self.holes.append((self._f.tell(), b))
            self._f.seek(n, 1)
            return n
        return self._f.readinto(b)


def _fill_holes(
    path: str, holes: Iterable[tuple[int, Any]], num_workers: int
) -> int:
    """Read the recorded payloads into their final memory. Returns request count.

    Slices are disjoint, so no locking is needed.
    """
    tasks = []
    for offset, buf in holes:
        mv = memoryview(buf).cast('B')
        for start in range(0, len(mv), _READ_CHUNK_SIZE):
            tasks.append((offset + start, mv[start : start + _READ_CHUNK_SIZE]))

    fd = os.open(path, os.O_RDONLY)

    def _pread(task: tuple[int, Any]) -> None:
        offset, mv = task
        done = 0
        while done < len(mv):
            # ``preadv`` may return fewer bytes than requested.
            got = os.preadv(fd, [mv[done:]], offset + done)
            if got <= 0:
                raise EOFError(
                    f"Unexpected end of file at offset {offset + done} of "
                    f"{path}, {len(mv) - done} bytes missing."
                )
            done += got

    try:
        # Iterating the map result waits for every task and re-raises the first
        # exception instead of dropping it.
        with ThreadPoolExecutor(num_workers) as pool:
            for _ in pool.map(_pread, tasks):
                pass
    finally:
        os.close(fd)

    return len(tasks)


def _fast_path_available(path, num_workers: int) -> bool:
    return (
        num_workers > 1
        # a BytesIO cannot be reopened by path and has no IO to parallelize
        and isinstance(path, str)
        # ``os.preadv`` is POSIX only
        and hasattr(os, 'preadv')
        # macOS splits the stream, see ``_pickle_loads_mac``
        and sys.platform != 'darwin'
    )


def parallel_safe_load_pickle(
    path, f, num_workers: int | None = None, encoding: str = 'latin1'
) -> Any:
    """Drop-in replacement for ``safe_load_pickle(f)`` with parallel payload reads.

    ``f`` is the caller's file object, used for the serial fallback. The result is
    structurally identical to the serial one, so downstream handling is unchanged.
    """
    num_workers = _resolve_num_workers(num_workers)
    if not _fast_path_available(path, num_workers):
        return safe_load_pickle(f, encoding=encoding)

    try:
        file_size = os.path.getsize(path)

        # Pass 1: parse the stream, skipping large payloads.
        with open(path, 'rb', buffering=_SCAN_BUFFER_SIZE) as scan_file:
            punched = _HolePunchFile(scan_file)
            result = safe_load_pickle(punched, encoding=encoding)
        holes = punched.holes

        if not holes:
            # protocol < 3, or tiny tensors only: nothing to parallelize.
            f.seek(0)
            return safe_load_pickle(f, encoding=encoding)

        for offset, buf in holes:
            if offset < 0 or offset + len(buf) > file_size:
                raise ValueError(
                    f"Payload out of range: {offset}+{len(buf)} exceeds the "
                    f"size {file_size} of {path}."
                )

        # Pass 2: fill the payloads in parallel.
        _fill_holes(path, holes, num_workers)
        return result
    except Exception as e:
        # Correctness first: drop the partial result and redo the load serially.
        warnings.warn(
            f"Parallel load of {path} failed ({type(e).__name__}: {e}), "
            "falling back to serial load."
        )
        f.seek(0)
        return safe_load_pickle(f, encoding=encoding)
