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

"""Tests for ``paddle.load(..., num_workers=n)`` parallel payload reading."""

from __future__ import annotations

import io
import os
import pickle
import sys
import tempfile
import unittest
import warnings

import numpy as np

import paddle
from paddle.framework import parallel_pickle_load as ppl

# one shape above the 1MB interception threshold, one below
BIG_SHAPE = [1024, 512]  # 2MB as float32
SMALL_SHAPE = [64]


def bitwise_equal(a, b):
    """Byte for byte comparison (NaN/-0.0 safe)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return (
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()
    )


def as_numpy(value):
    return value if isinstance(value, np.ndarray) else np.asarray(value)


class TestLoadNumWorkersAccuracy(unittest.TestCase):
    """Parallel path must reproduce the serial result bit for bit."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "ckpt.pdparams")

    def tearDown(self):
        self.temp_dir.cleanup()

    def check_roundtrip(self, state_dict, num_workers=8, **load_kwargs):
        paddle.save(state_dict, self.path)
        serial = paddle.load(self.path, return_numpy=True, **load_kwargs)
        parallel = paddle.load(
            self.path,
            return_numpy=True,
            num_workers=num_workers,
            **load_kwargs,
        )
        self.assertEqual(set(serial.keys()), set(parallel.keys()))
        for key in serial:
            self.assertTrue(
                bitwise_equal(as_numpy(serial[key]), as_numpy(parallel[key])),
                f"mismatch for key {key}",
            )
        return parallel

    def test_multiple_large_tensors(self):
        state_dict = {
            f"w{i}": paddle.uniform(BIG_SHAPE, dtype='float32')
            for i in range(4)
        }
        self.check_roundtrip(state_dict)

    def test_mixed_large_and_small_tensors(self):
        state_dict = {
            "big": paddle.uniform(BIG_SHAPE, dtype='float32'),
            "small": paddle.uniform(SMALL_SHAPE, dtype='float32'),
            "scalar": paddle.to_tensor(3.14, dtype='float32'),
        }
        self.check_roundtrip(state_dict)

    def test_dtypes(self):
        state_dict = {
            "fp32": paddle.uniform(BIG_SHAPE, dtype='float32'),
            "fp64": paddle.uniform(BIG_SHAPE, dtype='float64'),
            "fp16": paddle.uniform(BIG_SHAPE, dtype='float32').astype(
                'float16'
            ),
            "bf16": paddle.uniform(BIG_SHAPE, dtype='float32').astype(
                'bfloat16'
            ),
            "int64": paddle.arange(0, 1024 * 512, dtype='int64'),
            "int32": paddle.arange(0, 1024 * 512, dtype='int32'),
            "bool": paddle.arange(0, 1024 * 512, dtype='int32') % 2 == 0,
        }
        self.check_roundtrip(state_dict)

    def test_special_float_values(self):
        raw = np.full(BIG_SHAPE, 0.0, dtype='float32')
        raw[0, :4] = [np.nan, np.inf, -np.inf, -0.0]
        state_dict = {"special": paddle.to_tensor(raw)}
        loaded = self.check_roundtrip(state_dict)
        got = as_numpy(loaded["special"])
        self.assertTrue(np.isnan(got[0, 0]))
        self.assertEqual(got[0, 1], np.inf)
        self.assertEqual(got[0, 2], -np.inf)
        # -0.0 survives only if the bytes match
        self.assertEqual(np.signbit(got[0, 3]), True)

    def test_nested_structure(self):
        state_dict = {
            "list": [paddle.uniform(BIG_SHAPE), paddle.uniform(SMALL_SHAPE)],
            "tuple": (paddle.uniform(BIG_SHAPE),),
            "dict": {"inner": paddle.uniform(BIG_SHAPE)},
            "plain": 42,
            "text": "hello",
        }
        paddle.save(state_dict, self.path)
        serial = paddle.load(self.path, return_numpy=True)
        parallel = paddle.load(self.path, return_numpy=True, num_workers=8)
        self.assertEqual(serial["plain"], parallel["plain"])
        self.assertEqual(serial["text"], parallel["text"])
        for i in range(2):
            self.assertTrue(
                bitwise_equal(
                    as_numpy(serial["list"][i]), as_numpy(parallel["list"][i])
                )
            )
        self.assertTrue(
            bitwise_equal(
                as_numpy(serial["tuple"][0]), as_numpy(parallel["tuple"][0])
            )
        )
        self.assertTrue(
            bitwise_equal(
                as_numpy(serial["dict"]["inner"]),
                as_numpy(parallel["dict"]["inner"]),
            )
        )

    def test_layer_state_dict_and_names(self):
        """A real state_dict carries StructuredToParameterName@@."""
        layer = paddle.nn.Linear(512, 1024)
        paddle.save(layer.state_dict(), self.path)
        serial = paddle.load(self.path)
        parallel = paddle.load(self.path, num_workers=8)
        self.assertEqual(set(serial.keys()), set(parallel.keys()))
        for key in serial:
            self.assertTrue(
                bitwise_equal(
                    np.asarray(serial[key]), np.asarray(parallel[key])
                )
            )
            self.assertEqual(serial[key].name, parallel[key].name)

    def test_tensor_place_matches_serial(self):
        state_dict = {"w": paddle.uniform(BIG_SHAPE, dtype='float32')}
        paddle.save(state_dict, self.path)
        serial = paddle.load(self.path)
        parallel = paddle.load(self.path, num_workers=8)
        self.assertEqual(str(serial["w"].place), str(parallel["w"].place))
        self.assertTrue(paddle.equal_all(serial["w"], parallel["w"]))

    def test_worker_counts(self):
        state_dict = {
            f"w{i}": paddle.uniform(BIG_SHAPE, dtype='float32')
            for i in range(3)
        }
        paddle.save(state_dict, self.path)
        serial = paddle.load(self.path, return_numpy=True)
        for num_workers in (2, 3, 8, 33):
            parallel = paddle.load(
                self.path, return_numpy=True, num_workers=num_workers
            )
            for key in serial:
                self.assertTrue(
                    bitwise_equal(serial[key], parallel[key]),
                    f"mismatch with num_workers={num_workers}",
                )

    def test_repeated_load_is_stable(self):
        state_dict = {"w": paddle.uniform(BIG_SHAPE, dtype='float32')}
        paddle.save(state_dict, self.path)
        first = paddle.load(self.path, return_numpy=True, num_workers=8)["w"]
        for _ in range(3):
            again = paddle.load(self.path, return_numpy=True, num_workers=8)[
                "w"
            ]
            self.assertTrue(bitwise_equal(first, again))


class TestLoadNumWorkersArgumentCheck(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "ckpt.pdparams")
        paddle.save({"w": paddle.uniform(BIG_SHAPE)}, self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_type(self):
        for value in ("8", 3.5, True, [8]):
            with self.assertRaises(TypeError):
                paddle.load(self.path, num_workers=value)

    def test_negative(self):
        with self.assertRaises(ValueError):
            paddle.load(self.path, num_workers=-1)

    def test_unknown_config_still_rejected(self):
        with self.assertRaises(ValueError):
            paddle.load(self.path, no_such_option=1)

    def test_serial_values(self):
        expected = paddle.load(self.path, return_numpy=True)["w"]
        for value in (None, 0, 1):
            got = paddle.load(self.path, return_numpy=True, num_workers=value)[
                "w"
            ]
            self.assertTrue(bitwise_equal(expected, got))

    def test_resolve_num_workers(self):
        self.assertEqual(ppl._resolve_num_workers(None), 0)
        self.assertEqual(ppl._resolve_num_workers(0), 0)
        self.assertEqual(ppl._resolve_num_workers(16), 16)


class TestLoadNumWorkersFallback(unittest.TestCase):
    """Unsupported cases must silently produce the serial result."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "ckpt.pdparams")
        self.state_dict = {
            "big": paddle.uniform(BIG_SHAPE, dtype='float32'),
            "small": paddle.uniform(SMALL_SHAPE, dtype='float32'),
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def assert_matches_serial(self, **load_kwargs):
        serial = paddle.load(self.path, return_numpy=True)
        parallel = paddle.load(
            self.path, return_numpy=True, num_workers=8, **load_kwargs
        )
        for key in serial:
            self.assertTrue(bitwise_equal(serial[key], parallel[key]))

    def test_all_protocols_match_serial(self):
        """protocol 2 stores payloads as latin1 text (no interception, serial);
        protocol >= 3 stores raw bytes and is accelerated. Results must match."""
        for protocol in (2, 3, 4, 5):
            paddle.save(self.state_dict, self.path, protocol=protocol)
            with open(self.path, 'rb', buffering=8192) as f:
                punched = ppl._HolePunchFile(f)
                pickle.load(punched, encoding='latin1')
            if protocol == 2:
                self.assertEqual(
                    punched.holes, [], f"protocol {protocol} should not punch"
                )
            else:
                self.assertEqual(
                    len(punched.holes), 1, f"protocol {protocol} should punch"
                )
            self.assert_matches_serial()

    def test_only_small_tensors_falls_back(self):
        paddle.save(
            {f"t{i}": paddle.uniform(SMALL_SHAPE) for i in range(8)}, self.path
        )
        with open(self.path, 'rb', buffering=8192) as f:
            punched = ppl._HolePunchFile(f)
            pickle.load(punched, encoding='latin1')
        self.assertEqual(punched.holes, [])
        self.assert_matches_serial()

    def test_bytesio_input(self):
        buffer = io.BytesIO()
        paddle.save(self.state_dict, buffer)
        buffer.seek(0)
        serial = paddle.load(buffer, return_numpy=True)
        buffer.seek(0)
        parallel = paddle.load(buffer, return_numpy=True, num_workers=8)
        for key in serial:
            self.assertTrue(bitwise_equal(serial[key], parallel[key]))

    def test_missing_preadv(self):
        paddle.save(self.state_dict, self.path)
        preadv = os.preadv
        del os.preadv
        try:
            self.assertFalse(ppl._fast_path_available(self.path, 8))
            self.assert_matches_serial()
        finally:
            os.preadv = preadv

    def test_darwin_platform(self):
        paddle.save(self.state_dict, self.path)
        platform = sys.platform
        sys.platform = 'darwin'
        try:
            self.assertFalse(ppl._fast_path_available(self.path, 8))
        finally:
            sys.platform = platform

    def test_short_read_falls_back_with_warning(self):
        """A failing preadv must not leak a half filled tensor."""
        paddle.save(self.state_dict, self.path)
        serial = paddle.load(self.path, return_numpy=True)
        preadv = os.preadv
        os.preadv = lambda *args, **kwargs: 0
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                parallel = paddle.load(
                    self.path, return_numpy=True, num_workers=8
                )
            messages = [str(w.message) for w in caught]
        finally:
            os.preadv = preadv
        self.assertTrue(
            any("falling back to serial load" in m for m in messages),
            f"expected a fallback warning, got {messages}",
        )
        for key in serial:
            self.assertTrue(bitwise_equal(serial[key], parallel[key]))

    def test_out_of_range_payload_falls_back(self):
        paddle.save(self.state_dict, self.path)
        serial = paddle.load(self.path, return_numpy=True)
        original = ppl._HolePunchFile.readinto

        def poisoned(self, b):
            written = original(self, b)
            if self.holes:
                offset, buf = self.holes[-1]
                self.holes[-1] = (offset + (1 << 40), buf)
            return written

        ppl._HolePunchFile.readinto = poisoned
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                parallel = paddle.load(
                    self.path, return_numpy=True, num_workers=8
                )
            messages = [str(w.message) for w in caught]
        finally:
            ppl._HolePunchFile.readinto = original
        self.assertTrue(any("out of range" in m for m in messages), messages)
        for key in serial:
            self.assertTrue(bitwise_equal(serial[key], parallel[key]))


class TestHolePunchInternals(unittest.TestCase):
    """White box checks on the interception itself."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "ckpt.pdparams")

    def tearDown(self):
        self.temp_dir.cleanup()

    def scan(self, buffering=8192):
        with open(self.path, 'rb', buffering=buffering) as f:
            punched = ppl._HolePunchFile(f)
            obj = pickle.load(punched, encoding='latin1')
        return obj, punched.holes

    def test_one_payload_per_large_tensor(self):
        n_big = 3
        state_dict = {
            f"w{i}": paddle.uniform(BIG_SHAPE, dtype='float32')
            for i in range(n_big)
        }
        state_dict["small"] = paddle.uniform(SMALL_SHAPE)
        paddle.save(state_dict, self.path)
        _, holes = self.scan()
        self.assertEqual(len(holes), n_big)

    def test_payload_coverage_and_offsets(self):
        paddle.save(
            {"w": paddle.uniform([2048, 512], dtype='float32')}, self.path
        )
        size = os.path.getsize(self.path)
        _, holes = self.scan()
        covered = sum(len(buf) for _, buf in holes)
        # a small read buffer must leave (almost) nothing on the serial path
        self.assertGreater(covered / size, 0.99)
        for offset, buf in holes:
            self.assertGreaterEqual(offset, 0)
            self.assertLessEqual(offset + len(buf), size)

    def test_large_buffer_reduces_coverage(self):
        """Documents why the scan pass uses a small buffer."""
        paddle.save(
            {"w": paddle.uniform([2048, 512], dtype='float32')}, self.path
        )
        _, small_buffer = self.scan(buffering=8192)
        _, large_buffer = self.scan(buffering=1 << 20)
        self.assertGreater(
            sum(len(b) for _, b in small_buffer),
            sum(len(b) for _, b in large_buffer),
        )

    def test_scan_reads_only_metadata(self):
        """The scan pass must not transfer payload bytes."""
        paddle.save(
            {"w": paddle.uniform([2048, 512], dtype='float32')}, self.path
        )
        size = os.path.getsize(self.path)

        class CountingFile:
            def __init__(self, f):
                self._f = f
                self.read_bytes = 0

            def read(self, n=-1):
                data = self._f.read(n)
                self.read_bytes += len(data)
                return data

            def readline(self):
                data = self._f.readline()
                self.read_bytes += len(data)
                return data

            def peek(self, n=1):
                return self._f.peek(n)

            def seek(self, *args):
                return self._f.seek(*args)

            def tell(self):
                return self._f.tell()

            def readinto(self, b):
                got = self._f.readinto(b)
                self.read_bytes += got
                return got

        with open(self.path, 'rb', buffering=8192) as f:
            counting = CountingFile(f)
            punched = ppl._HolePunchFile(counting)
            pickle.load(punched, encoding='latin1')
        self.assertLess(counting.read_bytes, size // 100)

    def test_fill_holes_writes_file_content(self):
        paddle.save(
            {"w": paddle.uniform([2048, 512], dtype='float32')}, self.path
        )
        _, holes = self.scan()
        offset, buf = holes[0]
        ppl._fill_holes(self.path, holes, 4)
        with open(self.path, 'rb') as f:
            f.seek(offset)
            on_disk = f.read(len(buf))
        self.assertEqual(bytes(buf), on_disk)

    def test_fill_holes_request_count(self):
        payload = 4 * ppl._READ_CHUNK_SIZE
        elements = payload // 4
        paddle.save({"w": paddle.zeros([elements], dtype='float32')}, self.path)
        _, holes = self.scan()
        requests = ppl._fill_holes(self.path, holes, 4)
        self.assertGreaterEqual(requests, 4)

    def test_fill_holes_reports_missing_bytes(self):
        paddle.save({"w": paddle.uniform(BIG_SHAPE)}, self.path)
        _, holes = self.scan()
        preadv = os.preadv
        os.preadv = lambda *args, **kwargs: 0
        try:
            with self.assertRaises(EOFError):
                ppl._fill_holes(self.path, holes, 2)
        finally:
            os.preadv = preadv


class TestDcpLoadStateDictNumWorkers(unittest.TestCase):
    """``dist.load_state_dict`` must expose num_workers, defaulting to serial."""

    def test_signature_defaults(self):
        import inspect

        from paddle.distributed.flex_checkpoint.dcp import load_state_dict

        for fn in (
            load_state_dict.load_state_dict,
            load_state_dict.local_load_state_dict,
            load_state_dict.load_state_dict_impl,
        ):
            parameters = inspect.signature(fn).parameters
            self.assertIn("num_workers", parameters)
            self.assertEqual(parameters["num_workers"].default, 1)


if __name__ == '__main__':
    unittest.main()
