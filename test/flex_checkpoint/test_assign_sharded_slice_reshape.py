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
"""The copy layer of a reshape: ``assign_sharded_slice`` replaying a
postprocess list that holds reshape entries, and how it writes the target.

These call ``assign_sharded_slice`` with hand-made descriptors and
postprocess lists; the statements that produce them are tested in
``test_aoa_engine_reshape.py``.
"""

import unittest

import numpy as np

import paddle
from paddle.distributed.flex_checkpoint.aoa.aoa_engine import ShardedWeightDesc
from paddle.distributed.flex_checkpoint.aoa.postprocess import make_reshape_op
from paddle.distributed.flex_checkpoint.dcp.sharded_weight import ShardedWeight
from paddle.distributed.flex_checkpoint.dcp.utils import assign_sharded_slice

# canonical [E, H, 2, I] <-> flattened [E * H, 2 * I]
E, H, TWO, I = 2, 3, 2, 2
CANONICAL = (E, H, TWO, I)
FLATTENED = (E * H, TWO * I)
WHOLE_RESHAPE = make_reshape_op(CANONICAL)


def _arange(shape, dtype="float32"):
    return paddle.arange(int(np.prod(shape)), dtype=dtype).reshape(list(shape))


class TestAssignShardedSliceReshapeContract(unittest.TestCase):
    """What ``assign_sharded_slice`` checks when it replays a reshape entry.

    The entry names the local shape the slice takes, so the copy layer needs no
    geometry of its own; it only checks that the slice it holds can take that
    shape and then lands exactly on the destination region. A statement cannot
    get this wrong, a hand-written postprocess list can.
    """

    def _shards(self, source_shape, destination_shape):
        src_desc = ShardedWeightDesc(
            "s", source_shape, source_shape, (0,) * len(source_shape), "float32"
        )
        src_shard = ShardedWeight(
            "s",
            _arange(source_shape),
            source_shape,
            source_shape,
            (0,) * len(source_shape),
        )
        dst_desc = ShardedWeightDesc(
            "d",
            destination_shape,
            destination_shape,
            (0,) * len(destination_shape),
            "float32",
        )
        dst_shard = ShardedWeight(
            "d",
            paddle.zeros(list(destination_shape), dtype="float32"),
            destination_shape,
            destination_shape,
            (0,) * len(destination_shape),
        )
        return src_desc, src_shard, dst_desc, dst_shard

    def test_reshape_entry_with_another_element_count_is_refused(self):
        with self.assertRaisesRegex(ValueError, "element counts differ"):
            assign_sharded_slice(
                *self._shards((2, 4), CANONICAL), [WHOLE_RESHAPE]
            )

    def test_reshaped_slice_must_match_the_destination_region(self):
        # Same element count, other shape: the reshape itself is fine, but the
        # result would not land on the destination region.
        with self.assertRaisesRegex(AssertionError, "overlap shape mismatch"):
            assign_sharded_slice(
                *self._shards(FLATTENED, (E * H, TWO, I)), [WHOLE_RESHAPE]
            )


class TestAssignShardedSliceReshapePostprocess(unittest.TestCase):
    """Cover the transpose/cast postprocess loop that follows a reshape."""

    def _forward_src_dst(self, dtype):
        source = paddle.arange(
            int(np.prod(FLATTENED)), dtype="float32"
        ).reshape(list(FLATTENED))
        src_desc = ShardedWeightDesc(
            "s", FLATTENED, FLATTENED, (0, 0), "float32"
        )
        src_shard = ShardedWeight("s", source, FLATTENED, FLATTENED, (0, 0))
        out = paddle.zeros(list(CANONICAL), dtype=dtype)
        dst_desc = ShardedWeightDesc(
            "d", CANONICAL, CANONICAL, (0, 0, 0, 0), dtype
        )
        dst_shard = ShardedWeight("d", out, CANONICAL, CANONICAL, (0, 0, 0, 0))
        return source, src_desc, src_shard, dst_desc, dst_shard, out

    def test_reshape_then_transpose_then_cast(self):
        source, src_desc, src_shard, dst_desc, dst_shard, out = (
            self._forward_src_dst("float16")
        )
        permutation = [2, 1, 0, 3]
        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            [WHOLE_RESHAPE, str(permutation), "float16"],
        )
        expected = paddle.transpose(
            source.reshape(list(CANONICAL)), permutation
        ).astype("float16")
        np.testing.assert_allclose(out.numpy(), expected.numpy())

    def test_reshape_then_transpose(self):
        # perm [2, 1, 0, 3] keeps the shape (2, 3, 2, 2) unchanged.
        source, src_desc, src_shard, dst_desc, dst_shard, out = (
            self._forward_src_dst("float32")
        )
        assign_sharded_slice(
            src_desc,
            src_shard,
            dst_desc,
            dst_shard,
            [WHOLE_RESHAPE, "[2, 1, 0, 3]"],
        )
        expected = paddle.transpose(
            source.reshape(list(CANONICAL)), [2, 1, 0, 3]
        )
        np.testing.assert_allclose(out.numpy(), expected.numpy())


class TestAssignShardedSliceWriteMethod(unittest.TestCase):
    """Which write ``assign_sharded_slice`` uses, and that both land the data.

    A strided slice of a CPU target (load_state_dict stages a pinned optimizer
    state in a CPU copy) is written with ``target[index] = value``, which is
    several times faster there than ``paddle.assign`` into the slice. A region
    that does not share the target's buffer -- what paddle.slice returns with
    FLAGS_use_stride_kernel off -- has to be written that way, since assign
    would only fill the copy; anything whose value sits on another place than
    the target goes through assign.
    """

    class _Place:
        def __init__(self, cpu=False, xpu=False):
            self.cpu, self.xpu = cpu, xpu

        def is_cpu_place(self):
            return self.cpu

        def is_xpu_place(self):
            return self.xpu

    class _Tensor:
        def __init__(self, place, contiguous=True, shared=True):
            self.place, self.contiguous = place, contiguous
            self.shared = shared

        def is_contiguous(self):
            return self.contiguous

        def _is_shared_buffer_with(self, other):
            return self.shared

    def _decide(self, place, contiguous, shared=True):
        from paddle.distributed.flex_checkpoint.dcp.utils import (
            _write_by_setitem,
        )

        target = self._Tensor(place)
        region = self._Tensor(place, contiguous, shared)
        return _write_by_setitem(target, region, target)

    def test_cpu_strided_region_uses_setitem(self):
        place = self._Place(cpu=True)
        self.assertTrue(self._decide(place, contiguous=False))
        self.assertFalse(self._decide(place, contiguous=True))

    def test_a_region_that_is_a_copy_uses_setitem(self):
        # paddle.slice copies rather than views the target with
        # FLAGS_use_stride_kernel off (the XPU CI runs with it off), and
        # assign into that copy would leave the target unchanged.
        for place in (
            self._Place(cpu=True),
            self._Place(xpu=True),
            self._Place(),
        ):
            for contiguous in (True, False):
                self.assertTrue(self._decide(place, contiguous, shared=False))

    def test_other_devices_use_assign(self):
        place = self._Place()
        self.assertFalse(self._decide(place, contiguous=False))

    @unittest.skipUnless(
        paddle.is_compiled_with_cuda()
        and paddle.device.cuda.device_count() > 0,
        "needs a GPU",
    )
    def test_value_on_another_place_uses_assign(self):
        from paddle.distributed.flex_checkpoint.dcp.utils import (
            _write_by_setitem,
        )

        target = paddle.zeros([2, 3, 2, 2]).cpu()
        region = paddle.slice(target, axes=[2], starts=[0], ends=[1])
        value = paddle.ones([2, 3, 1, 2]).cuda()
        self.assertFalse(_write_by_setitem(target, region, value))

    def _write_half(self, postprocess_list, source_shape, target=None):
        """Fill the gate half (axis 2, index 0) of a CPU (2, 3, 2, 2) target."""
        target_shape = (2, 3, 2, 2)
        source = paddle.arange(12, dtype="float32").reshape(list(source_shape))
        source = source.cpu() + 1
        if target is None:
            target = paddle.zeros(list(target_shape), dtype="float32").cpu()
        pointer = target.data_ptr()
        zeros = (0,) * len(source_shape)
        assign_sharded_slice(
            ShardedWeightDesc(
                "s", source_shape, source_shape, zeros, "float32"
            ),
            ShardedWeight("s", source, source_shape, source_shape, zeros),
            ShardedWeightDesc(
                "d", (2, 3, 1, 2), target_shape, (0, 0, 0, 0), "float32"
            ),
            ShardedWeight("d", target, target_shape, target_shape, (0,) * 4),
            postprocess_list,
        )
        self.assertTrue(target.place.is_cpu_place())
        self.assertEqual(target.data_ptr(), pointer)
        return source, target

    def test_strided_cpu_write_without_reshape(self):
        source, target = self._write_half(None, (2, 3, 1, 2))
        np.testing.assert_array_equal(
            target.numpy()[:, :, 0:1, :], source.numpy()
        )
        np.testing.assert_array_equal(target.numpy()[:, :, 1:2, :], 0)

    def test_strided_cpu_write_after_a_reshape(self):
        source, target = self._write_half(
            [make_reshape_op((2, 3, 1, 2))], (6, 2)
        )
        np.testing.assert_array_equal(
            target.numpy()[:, :, 0, :], source.numpy().reshape(2, 3, 2)
        )
        np.testing.assert_array_equal(target.numpy()[:, :, 1, :], 0)

    def test_strided_cpu_write_into_a_parameter(self):
        # load_state_dict does not run under no_grad, and a parameter staged
        # on the CPU (``.cpu()`` of a pinned one) is a leaf that does not stop
        # gradient, which an in-place write refuses unless gradients are off.
        parameter = paddle.create_parameter(
            [2, 3, 2, 2],
            "float32",
            default_initializer=paddle.nn.initializer.Constant(0.0),
        ).cpu()
        self.assertFalse(parameter.stop_gradient)
        source, target = self._write_half(
            [make_reshape_op((2, 3, 1, 2))], (6, 2), target=parameter
        )
        np.testing.assert_array_equal(
            target.numpy()[:, :, 0, :], source.numpy().reshape(2, 3, 2)
        )
        np.testing.assert_array_equal(target.numpy()[:, :, 1, :], 0)


if __name__ == '__main__':
    unittest.main()
