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
outside a pickle frame, so ``_ParallelPayloadFile`` serves those requests with a
pool of ``os.preadv`` calls that write directly into the buffer the unpickler
provided. The data is in place before ``readinto`` returns, so the unpickler
never observes a partially filled buffer and nothing is assumed about the
lifetime of that memory. The file format is untouched and the fast path is
opt-in.
"""

from __future__ import annotations

import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .restricted_unpickler import safe_load_pickle

# Smaller payloads live inside pickle frames and are served from the frame
# buffer, so they never reach ``readinto``.
_MIN_PAYLOAD_SIZE = 1 << 20

# Bytes already prefetched by the buffered reader are served through ``read``
# and stay on the serial path, so keep the buffer small.
_BUFFER_SIZE = 8192

# Payloads are split so that a checkpoint with few tensors still keeps the pool
# busy. 32MB measured best (64MB limits parallelism, 16MB fragments IO).
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


def _pread_exact(fd: int, offset: int, mv: memoryview, path: str) -> None:
    """Read ``len(mv)`` bytes at ``offset``; ``preadv`` may return short."""
    done = 0
    while done < len(mv):
        got = os.preadv(fd, [mv[done:]], offset + done)
        if got <= 0:
            raise EOFError(
                f"Unexpected end of file at offset {offset + done} of {path}, "
                f"{len(mv) - done} bytes missing."
            )
        done += got


class _ParallelPayloadFile:
    """Serves large ``readinto`` requests with parallel ``preadv`` calls."""

    def __init__(self, f, fd: int, pool: ThreadPoolExecutor, path: str) -> None:
        self._f = f
        self._fd = fd
        self._pool = pool
        self._path = path
        self.payload_bytes = 0

    def read(self, n: int = -1) -> bytes:
        return self._f.read(n)

    def readline(self) -> bytes:
        return self._f.readline()

    def peek(self, n: int = 1) -> bytes:
        return self._f.peek(n)

    def readinto(self, b) -> int:
        n = len(b)
        if n < _MIN_PAYLOAD_SIZE:
            return self._f.readinto(b)
        offset = self._f.tell()
        mv = memoryview(b).cast('B')
        chunks = [
            (offset + start, mv[start : start + _READ_CHUNK_SIZE])
            for start in range(0, n, _READ_CHUNK_SIZE)
        ]
        # Iterating the map result waits for every chunk and re-raises the first
        # exception, so the buffer is complete once readinto returns.
        for _ in self._pool.map(
            lambda chunk: _pread_exact(self._fd, *chunk, self._path), chunks
        ):
            pass
        self._f.seek(n, 1)
        self.payload_bytes += n
        return n


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
        fd = os.open(path, os.O_RDONLY)
        try:
            with (
                ThreadPoolExecutor(num_workers) as pool,
                open(path, 'rb', buffering=_BUFFER_SIZE) as data_file,
            ):
                wrapped = _ParallelPayloadFile(data_file, fd, pool, path)
                return safe_load_pickle(wrapped, encoding=encoding)
        finally:
            os.close(fd)
    except Exception as e:
        # Correctness first: drop the partial result and redo the load serially.
        warnings.warn(
            f"Parallel load of {path} failed ({type(e).__name__}: {e}), "
            "falling back to serial load."
        )
        f.seek(0)
        return safe_load_pickle(f, encoding=encoding)
