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

import json
import os
import struct
import tempfile
import unittest
from dataclasses import dataclass
from unittest import mock

import numpy as np

import paddle
import paddle.distributed as dist
from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
    _apply_load_transform,
    _build_transform_component_load_dict,
    _load_checkpoint_data_file,
    _metadata_manager,
    _paddle_dtype,
    _target_shard_metadata,
)
from paddle.distributed.flex_checkpoint.dcp.metadata import (
    LocalTensorMetadata,
    Metadata,
)
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
    ShardedWeight,
    make_replicated_sharded_weight,
)
from paddle.distributed.flex_checkpoint.dcp.utils import (
    _safetensors_storage_dtype,
)

_QUANT_KEY = "weight.quant"
_SCALE_KEY = "weight.scale"


@dataclass(frozen=True)
class _ReadPlan:
    mode: str
    logical_local_shape: tuple[int, ...]
    logical_global_offset: tuple[int, ...]
    source_slices: dict[str, LocalTensorMetadata]


def _logical_metadata(dtype="float32"):
    return {
        "weight": LocalTensorMetadata(
            global_offset=(0, 0),
            local_shape=(4, 8),
            dtype=dtype,
            global_shape=(4, 8),
        )
    }


class _LocalTransform:
    """Transform that requests per-rank physical shards via read_plan()."""

    def __init__(self):
        self.plan = None
        self.source_key_calls = 0
        self.force_global_calls = 0

    def logical_metadata(self):
        return _logical_metadata()

    def source_keys(self, logical_key):
        self.source_key_calls += 1
        return [_QUANT_KEY, _SCALE_KEY]

    def read_plan(self, logical_key, target, force_global=False):
        if force_global:
            self.force_global_calls += 1
            self.plan = _ReadPlan("global", (4, 8), (0, 0), {})
        else:
            self.plan = _ReadPlan(
                "local",
                tuple(target.local_shape),
                tuple(target.global_offset),
                {
                    _QUANT_KEY: LocalTensorMetadata(
                        tuple(target.global_offset), (2, 8), "uint8", (4, 8)
                    ),
                    _SCALE_KEY: LocalTensorMetadata(
                        (1, 0), (1, 2), "uint8", (2, 2)
                    ),
                },
            )
        return self.plan

    def read_plan_for(self, logical_key):
        return self.plan

    def apply(self, logical_key, source_tensors, output_dtype):
        return source_tensors[_QUANT_KEY].astype(output_dtype)


class _LegacyTransform:
    """Transform without read_plan(): every source tensor is read globally."""

    def logical_metadata(self):
        return _logical_metadata()

    def source_keys(self, logical_key):
        return [_QUANT_KEY]

    def apply(self, logical_key, source_tensors, output_dtype):
        return source_tensors[_QUANT_KEY].astype(output_dtype)


class _BadDtypeTransform(_LegacyTransform):
    def logical_metadata(self):
        return _logical_metadata(dtype="not_a_dtype")


class _DequantTransform:
    """Assembles one logical fp32 tensor from a uint8 payload and a scale."""

    def logical_metadata(self):
        return _logical_metadata()

    def source_keys(self, logical_key):
        return [_QUANT_KEY, _SCALE_KEY]

    def apply(self, logical_key, source_tensors, output_dtype):
        payload = source_tensors[_QUANT_KEY].astype(output_dtype)
        scale = source_tensors[_SCALE_KEY].astype(output_dtype)
        return payload * scale


def _physical_metadata():
    return Metadata(
        state_dict_metadata={
            _QUANT_KEY: [LocalTensorMetadata((0, 0), (4, 8), "uint8", (4, 8))],
            _SCALE_KEY: [LocalTensorMetadata((0, 0), (2, 2), "uint8", (2, 2))],
        },
        storage_metadata={},
    )


def _target(local_shape=(2, 8), global_offset=(2, 0)):
    return ShardedWeight(
        key="weight",
        local_tensor=paddle.zeros(list(local_shape), dtype="float32"),
        local_shape=local_shape,
        global_shape=(4, 8),
        global_offset=global_offset,
    )


def _write_safetensors(path, entries):
    """Write a safetensors file directly so exotic dtypes can be produced.

    ``entries`` maps a tensor name to ``(storage_format, shape, raw_bytes)``.
    Formats such as ``F8_E4M3`` cannot be produced through numpy, so the
    container is assembled by hand: an 8-byte little-endian header length,
    a JSON header, then the concatenated payloads.
    """
    header = {}
    payload = b""
    for name, (storage_format, shape, raw) in entries.items():
        header[name] = {
            "dtype": storage_format,
            "shape": list(shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload += raw
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(payload)


def _fp8_checkpoint(directory, scale=2.0):
    """A HuggingFace-style checkpoint whose payload is raw one-byte fp8."""
    quant_bytes = bytes(range(32))
    _write_safetensors(
        os.path.join(directory, "model.safetensors"),
        {
            _QUANT_KEY: ("F8_E4M3", (4, 8), quant_bytes),
            _SCALE_KEY: (
                "F32",
                (1,),
                np.array([scale], dtype=np.float32).tobytes(),
            ),
        },
    )
    return np.frombuffer(quant_bytes, dtype=np.uint8).reshape(4, 8)


class TestLoadTransformComponentPlan(unittest.TestCase):
    def _build(self, logical_load_dict, transform, offload=False):
        return _build_transform_component_load_dict(
            logical_load_dict, _physical_metadata(), transform, offload
        )

    def test_returns_nothing_without_transform(self):
        self.assertEqual(self._build({"weight": _target()}, None), {})

    def test_builds_local_physical_components(self):
        components = self._build({"weight": _target()}, _LocalTransform())

        quant = components[_QUANT_KEY]
        scale = components[_SCALE_KEY]
        self.assertEqual(quant.global_shape, (4, 8))
        self.assertEqual(quant.local_shape, (2, 8))
        self.assertEqual(quant.global_offset, (2, 0))
        self.assertEqual(scale.global_shape, (2, 2))
        self.assertEqual(scale.local_shape, (1, 2))
        self.assertEqual(scale.global_offset, (1, 0))

    def test_preserves_component_offsets_in_tuple_keys(self):
        components = self._build(
            {("weight", (2, 0)): _target()}, _LocalTransform()
        )

        self.assertIn((_QUANT_KEY, (2, 0)), components)
        self.assertIn((_SCALE_KEY, (1, 0)), components)

    def test_forces_global_plan_for_mismatched_targets(self):
        transform = _LocalTransform()
        components = self._build(
            {
                ("weight", (0, 0)): _target(global_offset=(0, 0)),
                ("weight", (2, 0)): _target(global_offset=(2, 0)),
            },
            transform,
        )

        self.assertEqual(transform.force_global_calls, 1)
        # A global plan carries no source slices, so every component falls
        # back to a fully replicated read at offset zero.
        self.assertIn((_QUANT_KEY, (0, 0)), components)
        self.assertEqual(components[(_QUANT_KEY, (0, 0))].local_shape, (4, 8))
        self.assertEqual(components[(_SCALE_KEY, (0, 0))].local_shape, (2, 2))

    def test_offload_places_components_on_cpu(self):
        components = self._build(
            {"weight": _target()}, _LocalTransform(), offload=True
        )

        for component in components.values():
            self.assertTrue(component.local_tensor.place.is_cpu_place())

    def test_derives_metadata_from_plain_tensor_target(self):
        metadata = _target_shard_metadata(paddle.zeros([4, 8], dtype="float32"))

        self.assertEqual(metadata.global_shape, (4, 8))
        self.assertEqual(metadata.local_shape, (4, 8))
        self.assertEqual(metadata.global_offset, (0, 0))
        self.assertEqual(metadata.dtype, "float32")

    def test_derives_metadata_from_sharded_weight_target(self):
        metadata = _target_shard_metadata(_target())

        self.assertEqual(metadata.global_shape, (4, 8))
        self.assertEqual(metadata.local_shape, (2, 8))
        self.assertEqual(metadata.global_offset, (2, 0))
        self.assertEqual(metadata.dtype, "float32")


class TestApplyLoadTransform(unittest.TestCase):
    def _build(self, logical_load_dict, transform):
        return _build_transform_component_load_dict(
            logical_load_dict, _physical_metadata(), transform, False
        )

    def test_no_op_without_transform(self):
        target = _target()
        _apply_load_transform({"weight": target}, {}, None)

        np.testing.assert_array_equal(
            target.local_tensor.numpy(), np.zeros([2, 8], dtype="float32")
        )

    def test_collects_source_keys_once_per_logical_key(self):
        transform = _LocalTransform()
        self._build({"weight": _target()}, transform)

        self.assertEqual(transform.source_key_calls, 1)

    def test_assigns_local_transform_output_directly(self):
        transform = _LocalTransform()
        target = _target()
        components = self._build({"weight": target}, transform)
        components[_QUANT_KEY].local_tensor.fill_(1)

        _apply_load_transform({"weight": target}, components, transform)

        np.testing.assert_array_equal(
            target.local_tensor.numpy(), np.ones([2, 8], dtype="float32")
        )

    def test_legacy_transform_slices_global_output(self):
        transform = _LegacyTransform()
        target = _target()
        components = self._build({"weight": target}, transform)
        values = paddle.arange(32, dtype="float32").reshape([4, 8])
        paddle.assign(
            values.astype("uint8"), components[_QUANT_KEY].local_tensor
        )

        _apply_load_transform({"weight": target}, components, transform)

        # The target owns rows [2, 4) of the logical tensor.
        np.testing.assert_array_equal(
            target.local_tensor.numpy(), values.numpy()[2:]
        )

    def test_assigns_into_plain_tensor_target(self):
        transform = _LegacyTransform()
        target = paddle.zeros([4, 8], dtype="float32")
        components = self._build({"weight": target}, transform)
        values = paddle.arange(32, dtype="float32").reshape([4, 8])
        paddle.assign(
            values.astype("uint8"), components[_QUANT_KEY].local_tensor
        )

        _apply_load_transform({"weight": target}, components, transform)

        np.testing.assert_array_equal(target.numpy(), values.numpy())

    def test_copies_transform_output_across_places(self):
        transform = _LegacyTransform()
        target = paddle.zeros([4, 8], dtype="float32").cpu()
        components = self._build({"weight": target}, transform)
        components[_QUANT_KEY].local_tensor.fill_(3)

        _apply_load_transform({"weight": target}, components, transform)

        # The transform output lives on the default device; the target does
        # not, so it must be moved before the assign.
        self.assertTrue(target.place.is_cpu_place())
        np.testing.assert_array_equal(
            target.numpy(), np.full([4, 8], 3.0, dtype="float32")
        )

    def test_rejects_unsupported_logical_dtype(self):
        transform = _BadDtypeTransform()
        target = _target()
        components = self._build({"weight": target}, transform)

        with self.assertRaisesRegex(ValueError, "not_a_dtype"):
            _apply_load_transform({"weight": target}, components, transform)

    def test_paddle_dtype_accepts_prefixed_name(self):
        self.assertIs(_paddle_dtype("paddle.float32"), paddle.float32)
        self.assertIs(_paddle_dtype("uint8"), paddle.uint8)


class TestLoadTransformEndToEnd(unittest.TestCase):
    """Drives the public dist.load_state_dict(..., load_transform=...) API."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_path = self._tmp.name
        # A load that raises leaves the module-level metadata cache populated,
        # which would otherwise leak into the next test in this process.
        _metadata_manager.clear()

    def tearDown(self):
        self._tmp.cleanup()

    def _save_quantized_checkpoint(self, scale=2.0):
        payload = np.arange(32, dtype=np.uint8).reshape(4, 8)
        state_dict = {
            _QUANT_KEY: make_replicated_sharded_weight(
                _QUANT_KEY, paddle.to_tensor(payload)
            ),
            _SCALE_KEY: make_replicated_sharded_weight(
                _SCALE_KEY,
                paddle.to_tensor(np.array([scale], dtype=np.float32)),
            ),
        }
        dist.save_state_dict(state_dict, self.ckpt_path)
        return payload

    def test_load_state_dict_applies_transform(self):
        payload = self._save_quantized_checkpoint(scale=2.0)
        target = make_replicated_sharded_weight(
            "weight", paddle.zeros([4, 8], dtype="float32")
        )

        dist.load_state_dict(
            {"weight": target},
            self.ckpt_path,
            load_transform=_DequantTransform(),
        )

        np.testing.assert_allclose(
            target.local_tensor.numpy(),
            payload.astype("float32") * 2.0,
        )

    def test_rejects_component_key_colliding_with_target(self):
        self._save_quantized_checkpoint()
        state_dict = {
            "weight": make_replicated_sharded_weight(
                "weight", paddle.zeros([4, 8], dtype="float32")
            ),
            _QUANT_KEY: make_replicated_sharded_weight(
                _QUANT_KEY, paddle.zeros([4, 8], dtype="uint8")
            ),
        }

        with self.assertRaisesRegex(ValueError, _QUANT_KEY):
            dist.load_state_dict(
                state_dict,
                self.ckpt_path,
                load_transform=_DequantTransform(),
            )

    def test_load_state_dict_applies_transform_on_fp8_safetensors(self):
        payload = _fp8_checkpoint(self.ckpt_path, scale=2.0)
        target = make_replicated_sharded_weight(
            "weight", paddle.zeros([4, 8], dtype="float32")
        )

        dist.load_state_dict(
            {"weight": target},
            self.ckpt_path,
            safetensors=True,
            load_transform=_DequantTransform(),
        )

        # FlexCheckpoint must transport the fp8 bytes untouched; the transform
        # owns their numerical interpretation.
        np.testing.assert_allclose(
            target.local_tensor.numpy(),
            payload.astype("float32") * 2.0,
        )

    def test_aoa_renames_the_logical_key_of_a_transform(self):
        payload = self._save_quantized_checkpoint(scale=2.0)
        target = make_replicated_sharded_weight(
            "renamed", paddle.zeros([4, 8], dtype="float32")
        )

        dist.load_state_dict(
            {"renamed": target},
            self.ckpt_path,
            aoa_config={"aoa_statements": ["weight -> renamed"]},
            load_transform=_DequantTransform(),
        )

        # AOA sees the logical tensor published by the transform, not the
        # physical component keys it was assembled from.
        np.testing.assert_allclose(
            target.local_tensor.numpy(),
            payload.astype("float32") * 2.0,
        )

    def test_aoa_without_transform_loads_physical_keys(self):
        payload = self._save_quantized_checkpoint()
        target = make_replicated_sharded_weight(
            "quant", paddle.zeros([4, 8], dtype="uint8")
        )

        dist.load_state_dict(
            {"quant": target},
            self.ckpt_path,
            aoa_config={"aoa_statements": [f"{_QUANT_KEY} -> quant"]},
        )

        np.testing.assert_array_equal(target.local_tensor.numpy(), payload)

    def test_renamed_master_weight_reads_from_component_dict(self):
        """A passthrough target may be renamed by master-weight compatibility.

        ``get_rank_to_files`` rewrites the key inside the dict it is handed.
        With a transform that dict is a fresh merge of passthrough and
        component targets, so the restore loop can no longer find the renamed
        key in ``flat_state_dict`` and must fall back to it.
        """
        payload = np.arange(32, dtype=np.uint8).reshape(4, 8)
        master = np.arange(6, dtype=np.float32).reshape(2, 3)
        dist.save_state_dict(
            {
                _QUANT_KEY: make_replicated_sharded_weight(
                    _QUANT_KEY, paddle.to_tensor(payload)
                ),
                _SCALE_KEY: make_replicated_sharded_weight(
                    _SCALE_KEY,
                    paddle.to_tensor(np.array([2.0], dtype=np.float32)),
                ),
                "opt.w_fp32_master_0": make_replicated_sharded_weight(
                    "opt.w_fp32_master_0", paddle.to_tensor(master)
                ),
            },
            self.ckpt_path,
        )
        logical = make_replicated_sharded_weight(
            "weight", paddle.zeros([4, 8], dtype="float32")
        )
        # The checkpoint stores the flattened spelling; the target uses the
        # nested one.
        renamed = make_replicated_sharded_weight(
            "opt.master_weights.w", paddle.zeros([2, 3], dtype="float32")
        )

        dist.load_state_dict(
            {"weight": logical, "opt.master_weights.w": renamed},
            self.ckpt_path,
            load_transform=_DequantTransform(),
        )

        np.testing.assert_allclose(
            logical.local_tensor.numpy(), payload.astype("float32") * 2.0
        )
        np.testing.assert_allclose(renamed.local_tensor.numpy(), master)


class TestCheckpointDataFileLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _fp8_file(self):
        payload = _fp8_checkpoint(self.directory)
        return os.path.join(self.directory, "model.safetensors"), payload

    def test_transports_fp8_safetensors_as_lossless_uint8(self):
        path, payload = self._fp8_file()

        loaded = _load_checkpoint_data_file(path, True, False)

        self.assertEqual(loaded[_QUANT_KEY].dtype, paddle.uint8)
        np.testing.assert_array_equal(loaded[_QUANT_KEY].numpy(), payload)
        self.assertEqual(loaded[_SCALE_KEY].dtype, paddle.float32)

    def test_offload_puts_fp8_safetensors_on_cpu(self):
        path, payload = self._fp8_file()

        loaded = _load_checkpoint_data_file(path, True, True)

        self.assertEqual(loaded[_QUANT_KEY].dtype, paddle.uint8)
        self.assertTrue(loaded[_QUANT_KEY].place.is_cpu_place())
        np.testing.assert_array_equal(loaded[_QUANT_KEY].numpy(), payload)

    def test_restores_dtype_aliases_after_loading(self):
        path, _ = self._fp8_file()
        original = paddle.float8_e4m3fn
        had_numpy_alias = hasattr(np, "float8_e4m3fn")

        _load_checkpoint_data_file(path, True, False)

        self.assertIs(paddle.float8_e4m3fn, original)
        self.assertEqual(hasattr(np, "float8_e4m3fn"), had_numpy_alias)

    def test_restores_dtype_aliases_when_loading_fails(self):
        path, _ = self._fp8_file()
        original = paddle.float8_e4m3fn
        had_numpy_alias = hasattr(np, "float8_e4m3fn")

        def _fail(*args, **kwargs):
            raise RuntimeError("checkpoint read failed")

        with (
            mock.patch.object(paddle, "load", _fail),
            self.assertRaises(RuntimeError),
        ):
            _load_checkpoint_data_file(path, True, False)

        self.assertIs(paddle.float8_e4m3fn, original)
        self.assertEqual(hasattr(np, "float8_e4m3fn"), had_numpy_alias)

    def test_loads_paddle_format_checkpoint(self):
        path = os.path.join(self.directory, "shard.distcp")
        expected = np.arange(6, dtype=np.float32).reshape(2, 3)
        paddle.save({"w": paddle.to_tensor(expected)}, path)
        runs_on_cpu = paddle.get_device() == "cpu"

        for offload in (False, True):
            loaded = _load_checkpoint_data_file(path, False, offload)
            np.testing.assert_array_equal(loaded["w"].numpy(), expected)
            self.assertEqual(
                loaded["w"].place.is_cpu_place(),
                offload or runs_on_cpu,
            )


class TestSafetensorsStorageDtype(unittest.TestCase):
    def test_maps_registered_formats(self):
        self.assertEqual(_safetensors_storage_dtype("BF16"), "bfloat16")
        self.assertEqual(_safetensors_storage_dtype("F32"), "float32")
        self.assertEqual(_safetensors_storage_dtype("I64"), "int64")
        self.assertEqual(_safetensors_storage_dtype("BOOL"), "bool")

    def test_maps_raw_one_byte_formats_to_uint8(self):
        for storage_format in ("F8_E4M3", "F8_E4M3FN", "F8_E8M0"):
            self.assertEqual(
                _safetensors_storage_dtype(storage_format), "uint8"
            )

    def test_rejects_unregistered_format(self):
        with self.assertRaisesRegex(ValueError, "F4_E2M1"):
            _safetensors_storage_dtype("F4_E2M1")


if __name__ == "__main__":
    unittest.main()
