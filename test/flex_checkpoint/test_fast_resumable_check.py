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

import os
import pickle
import shutil
import struct
import tempfile
import unittest

import numpy as np

import paddle
from paddle.distributed.flex_checkpoint.dcp import (
    distcp_reader,
    metadata_reader,
)
from paddle.distributed.flex_checkpoint.dcp.distcp_reader import (
    scan_tensor_shapes,
)
from paddle.distributed.flex_checkpoint.dcp.fast_resumable import (
    check_resumable_locally_fast,
    is_unsharded_state_dict_metadata,
)
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorIndex,
    LocalTensorMetadata,
    Metadata,
)
from paddle.distributed.flex_checkpoint.dcp.metadata_manager import (
    MetadataManager,
)
from paddle.distributed.flex_checkpoint.dcp.metadata_reader import (
    load_state_dict_metadata,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp.utils import (
    check_resumable_locally,
)

# w_large is above pickle's 64 KiB frame target, so its payload is written
# outside the frame stream; w_small stays inside a frame. Both code paths of the
# header-only reader are therefore exercised.
SHAPES = {
    "w_large": [256, 256],
    "w_small": [4, 8],
    "bias": [8],
}

# A corrupt stream surfaces as whichever of these the pickle walk hits first.
CORRUPT_ERRORS = (
    EOFError,
    IndexError,
    KeyError,
    ValueError,
    struct.error,
    pickle.UnpicklingError,
)


def whole_tensor_meta(shape, dtype="float32"):
    """A LocalTensorMetadata for a tensor that is not split at all."""
    shape = tuple(shape)
    return LocalTensorMetadata(
        tuple([0] * len(shape)), shape, dtype, shape, False, None
    )


def build_metadata(shapes, num_fake_files=32):
    """A replicated Metadata: every fake rank stores every tensor whole.

    ``storage_metadata`` is deliberately made much bigger than
    ``state_dict_metadata``, the way it is on a real fully replicated
    checkpoint, so that skipping it is what the partial reader is tested on.
    """
    state_dict_metadata = {}
    storage_metadata = {}
    for key, shape in shapes.items():
        meta = whole_tensor_meta(shape)
        state_dict_metadata[key] = [meta]
        for rank in range(num_fake_files):
            index = LocalTensorIndex(
                tensor_key=key,
                global_offset=meta.global_offset,
                is_flattened=False,
                flattened_range=None,
                replica_id=rank,
                local_shape=meta.local_shape,
            )
            storage_metadata[index] = f"{rank}_0.distcp"
    return Metadata(state_dict_metadata, storage_metadata, {})


def build_deduped_metadata(layout):
    """A dedup'ed Metadata: `layout` maps a tensor key to (shape, file_name).

    This is what ``save_state_dict`` produces with ``save_replicas=False`` --
    every tensor is stored whole, but in exactly one rank's file.
    """
    state_dict_metadata = {}
    storage_metadata = {}
    for key, (shape, file_name) in layout.items():
        meta = whole_tensor_meta(shape)
        state_dict_metadata[key] = [meta]
        index = LocalTensorIndex(
            tensor_key=key,
            global_offset=meta.global_offset,
            is_flattened=False,
            flattened_range=None,
            replica_id=None,
            local_shape=meta.local_shape,
        )
        storage_metadata[index] = file_name
    return Metadata(state_dict_metadata, storage_metadata, {})


def build_state_dict(shapes):
    return {
        key: paddle.zeros(shape, dtype="float32")
        for key, shape in shapes.items()
    }


def write_bytes(path, data):
    with open(path, "wb") as f:
        f.write(data)
    return path


def push_str(name):
    """The protocol-4+ opcode sequence that pushes the short string `name`."""
    raw = name.encode("utf-8")
    return pickle.SHORT_BINUNICODE + bytes([len(raw)]) + raw


def one_frame(payload, declared_size=None):
    """A protocol-5 stream header plus a single FRAME wrapping `payload`."""
    size = len(payload) if declared_size is None else declared_size
    return (
        pickle.PROTO
        + bytes([5])
        + pickle.FRAME
        + struct.pack("<Q", size)
        + payload
    )


class ForbidFullLoad:
    """Fails the test if ``paddle.load`` is called while active."""

    def __enter__(self):
        self._orig = metadata_reader.paddle.load

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "paddle.load was called, the partial parse did not happen"
            )

        metadata_reader.paddle.load = forbidden
        return self

    def __exit__(self, *exc):
        metadata_reader.paddle.load = self._orig


class FakeAllGather:
    """Runs a ``use_dist=True`` check in one process with scripted verdicts.

    ``other_ranks`` is what the remaining ranks are pretending to report, so a
    single process can exercise the collective branch and check that exactly
    one collective is issued.
    """

    def __init__(self, other_ranks=()):
        self.other_ranks = list(other_ranks)
        self.calls = 0

    def __enter__(self):
        self._orig = paddle.distributed.all_gather_object

        def fake(object_list, obj, group=None):
            self.calls += 1
            object_list.extend([obj, *self.other_ranks])

        paddle.distributed.all_gather_object = fake
        return self

    def __exit__(self, *exc):
        paddle.distributed.all_gather_object = self._orig


class TestLoadStateDictMetadata(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_matches_full_load(self):
        path = os.path.join(self.tmp, "0.metadata")
        metadata = build_metadata(SHAPES)
        paddle.save(metadata, path)

        partial = load_state_dict_metadata(path)
        self.assertEqual(partial, metadata.state_dict_metadata)
        self.assertEqual(partial, paddle.load(path).state_dict_metadata)

    def test_storage_metadata_is_never_unpickled(self):
        """No ``paddle.load`` at all, and only a prefix of the file is read."""
        path = os.path.join(self.tmp, "0.metadata")
        metadata = build_metadata(SHAPES, num_fake_files=512)
        paddle.save(metadata, path)

        with ForbidFullLoad():
            partial = load_state_dict_metadata(path)
        self.assertEqual(partial, metadata.state_dict_metadata)

        with open(path, "rb") as f:
            cut = metadata_reader._scan_for_patterns(
                f, metadata_reader._key_patterns("storage_metadata")
            )
            needed = metadata_reader._bytes_needed(f, cut)
        self.assertGreater(cut, 0)
        self.assertLess(needed, os.path.getsize(path))

    def test_every_pickle_protocol(self):
        metadata = build_metadata(SHAPES)
        for protocol in (2, 4, 5):
            path = os.path.join(self.tmp, f"p{protocol}.metadata")
            paddle.save(metadata, path, protocol=protocol)
            with ForbidFullLoad():
                partial = load_state_dict_metadata(path)
            self.assertEqual(partial, metadata.state_dict_metadata, protocol)

    def test_scalar_and_empty_are_handled(self):
        path = os.path.join(self.tmp, "0.metadata")
        metadata = build_metadata({"scalar": []})
        paddle.save(metadata, path)
        self.assertEqual(
            load_state_dict_metadata(path), metadata.state_dict_metadata
        )

        path = os.path.join(self.tmp, "1.metadata")
        paddle.save(Metadata({}, {}, {}), path)
        self.assertEqual(load_state_dict_metadata(path), {})

    def test_declines_instead_of_full_load(self):
        path = os.path.join(self.tmp, "not_metadata")
        paddle.save({"a": paddle.zeros([2])}, path)
        self.assertIsNone(load_state_dict_metadata(path, allow_full_load=False))
        with self.assertRaises(ValueError):
            load_state_dict_metadata(path, allow_full_load=True)

    def test_full_load_fallback(self):
        """When the partial parse fails, the full load still answers."""
        path = os.path.join(self.tmp, "0.metadata")
        metadata = build_metadata(SHAPES)
        paddle.save(metadata, path)

        orig = metadata_reader._load_truncated
        metadata_reader._load_truncated = lambda p: None
        try:
            self.assertEqual(
                load_state_dict_metadata(path), metadata.state_dict_metadata
            )
            self.assertIsNone(
                load_state_dict_metadata(path, allow_full_load=False)
            )
        finally:
            metadata_reader._load_truncated = orig

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_state_dict_metadata(os.path.join(self.tmp, "nope.metadata"))

    def test_exported_as_public_api(self):
        self.assertIs(
            paddle.distributed.load_state_dict_metadata,
            load_state_dict_metadata,
        )
        self.assertIn("load_state_dict_metadata", paddle.distributed.__all__)


class TestMetadataReaderInternals(unittest.TestCase):
    """Directly covers the byte-level helpers of the partial reader."""

    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_field_after(self):
        self.assertEqual(
            metadata_reader._field_after("state_dict_metadata"),
            "storage_metadata",
        )
        self.assertIsNone(metadata_reader._field_after("flat_mapping"))
        self.assertIsNone(metadata_reader._field_after("not_a_field"))

    def test_key_patterns(self):
        short = metadata_reader._key_patterns("storage_metadata")
        self.assertEqual(len(short), 3)
        self.assertTrue(short[0].startswith(pickle.SHORT_BINUNICODE))
        for pattern in short:
            self.assertTrue(pattern.endswith(b"storage_metadata"))

        # A name of 256 bytes or more cannot use SHORT_BINUNICODE.
        long_name = "x" * 300
        long_patterns = metadata_reader._key_patterns(long_name)
        self.assertEqual(len(long_patterns), 2)
        for pattern in long_patterns:
            self.assertFalse(pattern.startswith(pickle.SHORT_BINUNICODE))

    def test_find_first(self):
        self.assertEqual(metadata_reader._find_first(b"--ab-a-", [b"a"]), 2)
        self.assertEqual(
            metadata_reader._find_first(b"--ab-a-", [b"ab", b"a"]), 2
        )
        self.assertEqual(
            metadata_reader._find_first(b"--ab-a-", [b"b", b"-a-"]), 3
        )
        self.assertEqual(metadata_reader._find_first(b"xyz", [b"a", b"b"]), -1)

    def test_scan_for_patterns_straddles_chunks(self):
        pattern = metadata_reader._key_patterns("storage_metadata")[0]
        offset = metadata_reader._SEARCH_CHUNK - 3
        path = write_bytes(
            os.path.join(self.tmp, "straddle.bin"),
            b"\x00" * offset + pattern + b"\x00" * 128,
        )
        with open(path, "rb") as f:
            self.assertEqual(
                metadata_reader._scan_for_patterns(f, [pattern]), offset
            )

    def test_scan_for_patterns_absent(self):
        path = write_bytes(os.path.join(self.tmp, "absent.bin"), b"\x00" * 4096)
        with open(path, "rb") as f:
            self.assertEqual(
                metadata_reader._scan_for_patterns(f, [b"nowhere"]), -1
            )

    def needed(self, name, data, cut):
        path = write_bytes(os.path.join(self.tmp, name), data)
        with open(path, "rb") as f:
            return metadata_reader._bytes_needed(f, cut)

    def test_bytes_needed_not_a_pickle(self):
        self.assertIsNone(self.needed("tiny.bin", b"x", 0))
        self.assertIsNone(self.needed("noproto.bin", b"ab" * 8, 0))

    def test_bytes_needed_unframed_protocol(self):
        data = pickle.PROTO + bytes([2]) + b"X" * 10
        self.assertEqual(self.needed("p2.bin", data, 5), 6)

    def test_bytes_needed_inside_frame(self):
        data = one_frame(b"X" * 10)
        self.assertEqual(self.needed("frame.bin", data, 15), len(data))

    def test_bytes_needed_rejects_broken_framing(self):
        # No FRAME opcode where one is required.
        self.assertIsNone(
            self.needed("p5raw.bin", pickle.PROTO + bytes([5]) + b"X" * 20, 10)
        )
        # A frame that declares a zero length.
        self.assertIsNone(
            self.needed("zero.bin", one_frame(b"X" * 10, declared_size=0), 12)
        )
        # A malformed header for the second frame.
        broken = one_frame(b"X" * 10) + b"Y" * 10
        self.assertIsNone(self.needed("broken2.bin", broken, 25))

    def test_bytes_needed_cut_outside_a_frame_payload(self):
        data = one_frame(b"X" * 10)
        # In the frame header, before the payload starts.
        self.assertIsNone(self.needed("before.bin", data, 5))
        # Past the end of the last frame.
        self.assertIsNone(self.needed("after.bin", data, 1000))

    def test_as_state_dict_metadata(self):
        as_sdm = metadata_reader._as_state_dict_metadata
        meta = whole_tensor_meta([4, 8])

        self.assertEqual(as_sdm({"w": [meta]}), {"w": [meta]})
        self.assertEqual(as_sdm({"w": (meta,)}), {"w": (meta,)})
        # A bare LocalTensorMetadata instead of a list of them.
        self.assertEqual(as_sdm({"w": meta}), {"w": meta})
        # The SETITEM layout: a partially filled Metadata.__dict__.
        self.assertEqual(
            as_sdm({"state_dict_metadata": {"w": [meta]}}), {"w": [meta]}
        )
        # An empty checkpoint is a valid answer, not a failure.
        self.assertEqual(as_sdm({}), {})

        self.assertIsNone(as_sdm([meta]))
        self.assertIsNone(as_sdm({"flat_mapping": {}}))
        self.assertIsNone(as_sdm({1: [meta]}))
        self.assertIsNone(as_sdm({"w": 5}))
        self.assertIsNone(as_sdm({"w": [meta, "not a meta"]}))

    def test_load_truncated_declines_when_field_order_differs(self):
        """The cut marker is there, but it does not end state_dict_metadata."""
        path = write_bytes(
            os.path.join(self.tmp, "wrong_order.metadata"),
            pickle.dumps({"storage_metadata": 1}, protocol=5),
        )
        self.assertIsNone(metadata_reader._load_truncated(path))

    def test_load_truncated_declines_on_unparsable_prefix(self):
        payload = (
            push_str("state_dict_metadata")
            + b"\xff"  # not a pickle opcode
            + push_str("storage_metadata")
        )
        data = one_frame(payload)
        path = write_bytes(os.path.join(self.tmp, "bad_opcode.metadata"), data)
        self.assertGreater(data.find(push_str("storage_metadata")), 0)
        self.assertIsNone(metadata_reader._load_truncated(path))

    def test_load_truncated_declines_without_a_cut_field(self):
        path = os.path.join(self.tmp, "0.metadata")
        paddle.save(build_metadata(SHAPES), path)

        orig = metadata_reader._WANTED_FIELD
        metadata_reader._WANTED_FIELD = metadata_reader._FIELDS[-1]
        try:
            self.assertIsNone(metadata_reader._load_truncated(path))
        finally:
            metadata_reader._WANTED_FIELD = orig

    def test_load_truncated_declines_on_unusable_cut(self):
        path = os.path.join(self.tmp, "0.metadata")
        paddle.save(build_metadata(SHAPES), path)

        orig = metadata_reader._bytes_needed
        try:
            # The framing could not be walked.
            metadata_reader._bytes_needed = lambda f, cut: None
            self.assertIsNone(metadata_reader._load_truncated(path))
            # The cut is not inside the prefix that has to be read.
            metadata_reader._bytes_needed = lambda f, cut: cut
            self.assertIsNone(metadata_reader._load_truncated(path))
            # The file is shorter than the prefix, e.g. it shrank under us.
            metadata_reader._bytes_needed = lambda f, cut: 1 << 30
            self.assertIsNone(metadata_reader._load_truncated(path))
        finally:
            metadata_reader._bytes_needed = orig

    def test_load_truncated_declines_when_marker_is_absent(self):
        path = write_bytes(
            os.path.join(self.tmp, "no_marker.metadata"),
            pickle.dumps({"a": 1}, protocol=5),
        )
        self.assertIsNone(metadata_reader._load_truncated(path))


class TestScanTensorShapes(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def save_and_scan(self, state_dict, name="0_0.distcp"):
        path = os.path.join(self.tmp, name)
        paddle.save(state_dict, path)
        return scan_tensor_shapes(path)

    def test_shapes_match_paddle_load(self):
        path = os.path.join(self.tmp, "0_0.distcp")
        paddle.save(build_state_dict(SHAPES), path)

        scanned = scan_tensor_shapes(path)
        truth = paddle.load(path, return_numpy=True)
        for key, shape in SHAPES.items():
            self.assertIn(key, scanned)
            self.assertEqual(scanned[key], tuple(shape))
            self.assertEqual(scanned[key], tuple(truth[key].shape))

    def test_payload_contents_do_not_matter(self):
        filled = {
            key: paddle.to_tensor(
                np.arange(int(np.prod(shape)), dtype="float32").reshape(shape)
            )
            for key, shape in SHAPES.items()
        }
        self.assertEqual(
            self.save_and_scan(filled),
            {key: tuple(shape) for key, shape in SHAPES.items()},
        )

    def test_every_dtype(self):
        dtypes = [
            "float32",
            "float16",
            "bfloat16",
            "int64",
            "int32",
            "uint8",
            "bool",
        ]
        state_dict = {
            dtype: paddle.zeros([3, 5], dtype=dtype) for dtype in dtypes
        }
        scanned = self.save_and_scan(state_dict)
        self.assertEqual(scanned, dict.fromkeys(dtypes, (3, 5)))

    def test_scalar_and_empty_shapes(self):
        scanned = self.save_and_scan(
            {
                "scalar": paddle.zeros([], dtype="float32"),
                "empty": paddle.zeros([0, 4], dtype="float32"),
            }
        )
        self.assertEqual(scanned, {"scalar": (), "empty": (0, 4)})

    def test_around_the_frame_size_target(self):
        """float32 lengths whose payloads bracket pickle's 64 KiB target."""
        numels = (16383, 16384, 16385)
        state_dict = {
            f"w{numel}": paddle.zeros([numel], dtype="float32")
            for numel in numels
        }
        scanned = self.save_and_scan(state_dict)
        self.assertEqual(scanned, {f"w{numel}": (numel,) for numel in numels})

    def test_non_tensor_entries_are_skipped(self):
        scanned = self.save_and_scan(
            {
                "w": paddle.zeros([4, 8], dtype="float32"),
                "StructuredToParameterName@@": {"w": "w"},
            }
        )
        self.assertEqual(scanned, {"w": (4, 8)})

    def test_truncated_file_raises(self):
        path = os.path.join(self.tmp, "0_0.distcp")
        paddle.save(build_state_dict(SHAPES), path)
        cut = os.path.getsize(path) // 2
        truncated = os.path.join(self.tmp, "1_0.distcp")
        with open(path, "rb") as src:
            head = src.read(cut)
        write_bytes(truncated, head)
        # Where the cut lands decides which of these the walk hits first.
        with self.assertRaises(CORRUPT_ERRORS):
            scan_tensor_shapes(truncated)

    def test_not_a_pickle_raises(self):
        path = write_bytes(
            os.path.join(self.tmp, "junk.distcp"), b"\xffnot a pickle"
        )
        with self.assertRaises(CORRUPT_ERRORS):
            scan_tensor_shapes(path)

    def test_non_dict_top_level_raises(self):
        path = write_bytes(
            os.path.join(self.tmp, "list.distcp"),
            pickle.dumps([1, 2, 3], protocol=5),
        )
        with self.assertRaises(ValueError):
            scan_tensor_shapes(path)

    def test_tensor_shape_rejects_foreign_values(self):
        shape_of = distcp_reader._tensor_shape
        payload = distcp_reader._Payload(16)
        frombuffer = "numpy.core.numeric._frombuffer"

        self.assertEqual(
            shape_of(distcp_reader._Call(frombuffer, (payload, "f4", (2, 2)))),
            (2, 2),
        )
        self.assertIsNone(shape_of(None))
        self.assertIsNone(shape_of(distcp_reader._Call("builtins.tuple", ())))
        # Right global, but not the argument list of a pickled ndarray.
        self.assertIsNone(shape_of(distcp_reader._Call(frombuffer, (payload,))))
        self.assertIsNone(
            shape_of(distcp_reader._Call(frombuffer, ("raw", "f4", (2, 2))))
        )
        self.assertIsNone(
            shape_of(distcp_reader._Call(frombuffer, (payload, "f4", 4)))
        )


class TestIsUnshardedStateDictMetadata(unittest.TestCase):
    def test_unsharded(self):
        self.assertTrue(
            is_unsharded_state_dict_metadata(
                {
                    "w": [whole_tensor_meta([16, 4])],
                    "scalar": [whole_tensor_meta([])],
                }
            )
        )

    def test_empty_checkpoint(self):
        self.assertTrue(is_unsharded_state_dict_metadata({}))

    def test_bare_metadata_instead_of_a_list(self):
        self.assertTrue(
            is_unsharded_state_dict_metadata({"w": whole_tensor_meta([16, 4])})
        )
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": LocalTensorMetadata(
                        (8, 0), (8, 4), "float32", (16, 4), False, None
                    )
                }
            )
        )

    def test_tuple_of_one_is_accepted(self):
        self.assertTrue(
            is_unsharded_state_dict_metadata({"w": (whole_tensor_meta([4]),)})
        )

    def test_unknown_global_shape_is_accepted(self):
        """``global_shape=None`` cannot contradict the local shape."""
        self.assertTrue(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (0,), (4,), "float32", None, False, None
                        )
                    ]
                }
            )
        )

    def test_two_shards(self):
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (0, 0), (8, 4), "float32", (16, 4), False, None
                        ),
                        LocalTensorMetadata(
                            (8, 0), (8, 4), "float32", (16, 4), False, None
                        ),
                    ]
                }
            )
        )

    def test_no_shard_at_all(self):
        self.assertFalse(is_unsharded_state_dict_metadata({"w": []}))

    def test_nonzero_global_offset(self):
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (8, 0), (8, 4), "float32", (16, 4), False, None
                        )
                    ]
                }
            )
        )

    def test_flattened(self):
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (0,), (32,), "float32", (32,), True, (0, 32)
                        )
                    ]
                }
            )
        )

    def test_flattened_range_without_the_flag(self):
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (0,), (32,), "float32", (32,), False, (0, 16)
                        )
                    ]
                }
            )
        )

    def test_local_shape_differs_from_global(self):
        self.assertFalse(
            is_unsharded_state_dict_metadata(
                {
                    "w": [
                        LocalTensorMetadata(
                            (0, 0), (8, 4), "float32", (16, 4), False, None
                        )
                    ]
                }
            )
        )


class TestCheckResumableLocallyFast(unittest.TestCase):
    """The fast check must agree with utils.check_resumable_locally."""

    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()
        self.metadata_path = os.path.join(self.tmp, "0.metadata")
        self.metadata = build_metadata(SHAPES)
        paddle.save(self.metadata, self.metadata_path)
        paddle.save(
            build_state_dict(SHAPES), os.path.join(self.tmp, "0_0.distcp")
        )
        self.manager = MetadataManager()
        self.manager.set_metadata_list([self.metadata])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fast(self, state_dict, path=None):
        return check_resumable_locally_fast(
            self.metadata_path,
            self.tmp if path is None else path,
            state_dict,
            use_dist=False,
        )

    def stock(self, state_dict, path=None):
        return check_resumable_locally(
            self.tmp if path is None else path,
            state_dict,
            self.manager,
            False,
            None,
        )

    def assert_agree(self, state_dict, expected, path=None):
        fast = self.fast(state_dict, path)
        stock = self.stock(state_dict, path)
        self.assertEqual(fast, expected)
        self.assertEqual(stock, expected)

    def test_resumable(self):
        self.assert_agree(build_state_dict(SHAPES), True)

    def test_empty_state_dict(self):
        self.assert_agree({}, True)

    def test_subset_of_the_checkpoint_is_resumable(self):
        state_dict = build_state_dict(SHAPES)
        state_dict.pop("bias")
        self.assert_agree(state_dict, True)

    def test_key_absent_from_checkpoint(self):
        state_dict = build_state_dict(SHAPES)
        state_dict["not.in.checkpoint"] = paddle.zeros([3], dtype="float32")
        self.assert_agree(state_dict, False)

    def test_shape_mismatch(self):
        state_dict = build_state_dict(SHAPES)
        state_dict["bias"] = paddle.zeros([7], dtype="float32")
        self.assert_agree(state_dict, False)

    def test_dtype_mismatch_is_tolerated(self):
        """Like the stock check, only shapes are compared."""
        state_dict = build_state_dict(SHAPES)
        state_dict["bias"] = paddle.zeros([8], dtype="float16")
        self.assert_agree(state_dict, True)

    def test_uninitialized_tensor_is_ignored(self):
        """Both checks skip a tensor that has no metadata to compare."""
        state_dict = build_state_dict(SHAPES)
        state_dict["not_allocated"] = paddle.Tensor()
        self.assertFalse(state_dict["not_allocated"]._is_initialized())
        self.assert_agree(state_dict, True)

    def test_checkpoint_file_missing(self):
        empty = tempfile.mkdtemp(dir=self.tmp)
        self.assert_agree(build_state_dict(SHAPES), False, path=empty)

    def test_sharded_weight_stored_whole(self):
        state_dict = {
            "w_small": ShardedWeight(
                key="w_small",
                local_tensor=paddle.zeros([4, 8], dtype="float32"),
                local_shape=(4, 8),
                global_shape=(4, 8),
                global_offset=(0, 0),
            )
        }
        self.assert_agree(state_dict, True)

    def test_sharded_weight_with_an_offset(self):
        state_dict = {
            "w_small": ShardedWeight(
                key="w_small",
                local_tensor=paddle.zeros([4, 8], dtype="float32"),
                local_shape=(4, 8),
                global_shape=(8, 8),
                global_offset=(4, 0),
            )
        }
        self.assert_agree(state_dict, False)

    def test_flattened_sharded_weight(self):
        state_dict = {
            "bias": ShardedWeight(
                key="bias",
                local_tensor=paddle.zeros([8], dtype="float32"),
                local_shape=(8,),
                global_shape=(8,),
                global_offset=(0,),
                is_flattened=True,
                flattened_range=slice(0, 8),
            )
        }
        self.assert_agree(state_dict, False)

    def test_declines_when_checkpoint_is_sharded(self):
        sharded = Metadata(
            {
                "w": [
                    LocalTensorMetadata(
                        (0, 0), (8, 4), "float32", (16, 4), False, None
                    ),
                    LocalTensorMetadata(
                        (8, 0), (8, 4), "float32", (16, 4), False, None
                    ),
                ]
            },
            {},
            {},
        )
        path = os.path.join(self.tmp, "sharded.metadata")
        paddle.save(sharded, path)
        self.assertIsNone(
            check_resumable_locally_fast(
                path, self.tmp, build_state_dict(SHAPES), use_dist=False
            )
        )

    def test_declines_when_metadata_missing(self):
        self.assertIsNone(
            check_resumable_locally_fast(
                os.path.join(self.tmp, "nope.metadata"),
                self.tmp,
                build_state_dict(SHAPES),
                use_dist=False,
            )
        )

    def test_declines_when_metadata_is_unparsable(self):
        """A file that exists but is not a partially readable Metadata."""
        self.assertIsNone(
            check_resumable_locally_fast(
                os.path.join(self.tmp, "0_0.distcp"),
                self.tmp,
                build_state_dict(SHAPES),
                use_dist=False,
            )
        )

    def test_truncated_checkpoint_file_is_rejected(self):
        """The stock check only stats the file; the fast one reads its header."""
        path = tempfile.mkdtemp(dir=self.tmp)
        source = os.path.join(self.tmp, "0_0.distcp")
        with open(source, "rb") as src:
            head = src.read(os.path.getsize(source) // 2)
        write_bytes(os.path.join(path, "0_0.distcp"), head)
        state_dict = build_state_dict(SHAPES)
        self.assertFalse(self.fast(state_dict, path))
        self.assertTrue(self.stock(state_dict, path))


class TestFastCheckCollective(unittest.TestCase):
    """The distributed branch: one all_gather_object, and only when it applies.

    ``all_gather_object`` is replaced so a single process can play a whole world,
    which is what makes the collective count observable.
    """

    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()
        # use_dist=True makes both checks look for this rank's own file.
        self.rank = paddle.distributed.get_rank()
        self.metadata_path = os.path.join(self.tmp, "0.metadata")
        self.metadata = build_metadata(
            SHAPES, num_fake_files=max(32, self.rank + 1)
        )
        paddle.save(self.metadata, self.metadata_path)
        paddle.save(
            build_state_dict(SHAPES),
            os.path.join(self.tmp, f"{self.rank}_0.distcp"),
        )
        self.manager = MetadataManager()
        self.manager.set_metadata_list([self.metadata])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fast(self, other_ranks, path=None):
        with FakeAllGather(other_ranks) as gather:
            verdict = check_resumable_locally_fast(
                self.metadata_path,
                self.tmp if path is None else path,
                build_state_dict(SHAPES),
                use_dist=True,
            )
        return verdict, gather.calls

    def stock(self, other_ranks, path=None):
        with FakeAllGather(other_ranks) as gather:
            verdict = check_resumable_locally(
                self.tmp if path is None else path,
                build_state_dict(SHAPES),
                self.manager,
                True,
                None,
            )
        return verdict, gather.calls

    def test_all_ranks_agree(self):
        self.assertEqual(self.fast([True, True]), (True, 1))
        self.assertEqual(self.stock([True, True]), (True, 1))

    def test_one_rank_dissents(self):
        self.assertEqual(self.fast([True, False]), (False, 1))
        self.assertEqual(self.stock([True, False]), (False, 1))

    def test_this_rank_cannot_resume(self):
        empty = tempfile.mkdtemp(dir=self.tmp)
        self.assertEqual(self.fast([True, True], path=empty), (False, 1))
        self.assertEqual(self.stock([True, True], path=empty), (False, 1))

    def test_no_collective_when_the_fast_path_declines(self):
        sharded = Metadata(
            {"w": [whole_tensor_meta([4]), whole_tensor_meta([4])]}, {}, {}
        )
        self.metadata_path = os.path.join(self.tmp, "sharded.metadata")
        paddle.save(sharded, self.metadata_path)
        self.assertEqual(self.fast([True, True]), (None, 0))


class TestDedupedCheckpoint(unittest.TestCase):
    """A tensor that lives in another rank's file must not be claimed as local.

    With ``save_replicas=False`` a replicated tensor is written by exactly one
    rank, so rank 0's file holds only part of the checkpoint. The fast check
    must reject a state_dict asking for a tensor stored elsewhere, which is what
    keeps "the key is in my file" a sound identity test: the key is only ever in
    the file that ``storage_metadata`` assigns it to.
    """

    def setUp(self):
        paddle.set_device("cpu")
        self.tmp = tempfile.mkdtemp()
        self.metadata = build_deduped_metadata(
            {
                "mine": ([16, 4], "0_0.distcp"),
                "theirs": ([8, 2], "1_0.distcp"),
            }
        )
        self.metadata_path = os.path.join(self.tmp, "0.metadata")
        paddle.save(self.metadata, self.metadata_path)
        paddle.save(
            build_state_dict({"mine": [16, 4]}),
            os.path.join(self.tmp, "0_0.distcp"),
        )
        paddle.save(
            build_state_dict({"theirs": [8, 2]}),
            os.path.join(self.tmp, "1_0.distcp"),
        )
        self.manager = MetadataManager()
        self.manager.set_metadata_list([self.metadata])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assert_agree(self, shapes, expected):
        state_dict = build_state_dict(shapes)
        fast = check_resumable_locally_fast(
            self.metadata_path, self.tmp, state_dict, use_dist=False
        )
        stock = check_resumable_locally(
            self.tmp, state_dict, self.manager, False, None
        )
        self.assertEqual(fast, expected)
        self.assertEqual(stock, expected)

    def test_own_share_is_resumable(self):
        self.assert_agree({"mine": [16, 4]}, True)

    def test_tensor_owned_by_another_rank(self):
        self.assert_agree({"mine": [16, 4], "theirs": [8, 2]}, False)


if __name__ == "__main__":
    unittest.main()
