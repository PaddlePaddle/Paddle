#   Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
With ``FLAGS_pinned_memory_as_cpu_backend`` on, ops on CUDA-pinned tensors
run CPU kernels. The pinned input must not be copied to GPU and shared back
into itself: it has to stay on pinned memory, and view ops (reshape / slice /
view) must return views on the same pinned buffer.
"""

import inspect
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
    _pinned_memory_as_cpu_backend,
)

FLAG = "FLAGS_pinned_memory_as_cpu_backend"


def _pinned(shape):
    numel = int(np.prod(shape))
    return paddle.arange(numel, dtype="float32").reshape(shape).pin_memory()


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "CUDA pinned memory is required"
)
class TestPinnedMemoryAsCpuBackend(unittest.TestCase):
    def assertPinned(self, t):
        self.assertTrue(t.place.is_cuda_pinned_place(), str(t.place))

    def test_reshape_keeps_input_pinned_and_shares_storage(self):
        with _pinned_memory_as_cpu_backend():
            x = _pinned([6, 4, 2])
            ptr, ref = x.data_ptr(), x.numpy().copy()
            y = x.reshape([24, 2])
            self.assertPinned(x)
            self.assertEqual(x.data_ptr(), ptr)
            self.assertPinned(y)
            self.assertEqual(y.data_ptr(), ptr)
            np.testing.assert_array_equal(y.numpy().ravel(), ref.ravel())

    def test_slice_keeps_input_pinned(self):
        with _pinned_memory_as_cpu_backend():
            x = _pinned([16])
            ptr = x.data_ptr()
            y = paddle.slice(x, axes=[0], starts=[2], ends=[10])
            self.assertPinned(x)
            self.assertEqual(x.data_ptr(), ptr)
            self.assertPinned(y)
            np.testing.assert_array_equal(y.numpy(), x.numpy()[2:10])

    def test_view_shares_storage(self):
        with _pinned_memory_as_cpu_backend():
            x = _pinned([8, 4])
            ptr = x.data_ptr()
            y = x.view([4, 8])
            self.assertPinned(x)
            self.assertEqual(y.data_ptr(), ptr)

    def test_flag_restored(self):
        old = paddle.get_flags([FLAG])[FLAG]
        with _pinned_memory_as_cpu_backend():
            self.assertTrue(paddle.get_flags([FLAG])[FLAG])
        self.assertEqual(paddle.get_flags([FLAG])[FLAG], old)

    def test_flag_restored_on_exception(self):
        old = paddle.get_flags([FLAG])[FLAG]
        with self.assertRaises(RuntimeError), _pinned_memory_as_cpu_backend():
            raise RuntimeError("boom")
        self.assertEqual(paddle.get_flags([FLAG])[FLAG], old)

    def test_default_behaviour_unchanged(self):
        # Outside the block a pinned input still selects the GPU kernel.
        x = _pinned([8])
        y = x + 1
        self.assertTrue(y.place.is_gpu_place(), str(y.place))

    def test_gpu_tensor_unaffected(self):
        with _pinned_memory_as_cpu_backend():
            x = paddle.ones([4, 4]).cuda()
            y = x.reshape([16])
        self.assertTrue(x.place.is_gpu_place())
        self.assertTrue(y.place.is_gpu_place())


class TestMuonShardedStateDictGuard(unittest.TestCase):
    """MuonShardingOptimizer.sharded_state_dict() must run its body with the
    pinned-as-CPU flag on and restore it afterwards. Driving the real
    optimizer needs a distributed Muon setup, so the wrapper is called on a
    stand-in that records the flag its body saw.
    """

    def test_body_runs_with_flag_and_restores(self):
        from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
            MuonShardingOptimizer,
        )

        seen = {}

        class Stub:
            def _sharded_state_dict_impl(self, model_sharded_state_dict):
                seen["flag"] = paddle.get_flags([FLAG])[FLAG]
                seen["arg"] = model_sharded_state_dict
                return {"ok": True}

        before = paddle.get_flags([FLAG])[FLAG]
        out = MuonShardingOptimizer.sharded_state_dict(Stub(), {"w": 1})
        self.assertEqual(out, {"ok": True})
        self.assertTrue(seen["flag"])
        self.assertEqual(seen["arg"], {"w": 1})
        self.assertEqual(paddle.get_flags([FLAG])[FLAG], before)


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "CUDA pinned memory is required"
)
class TestStagePinnedTargetsOnCpu(unittest.TestCase):
    """flex_checkpoint load stages pinned targets on CPU and copies the result
    back, so the pinned tensors keep their place and storage."""

    def _stage(self):
        from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
            _stage_pinned_targets_on_cpu,
        )

        return _stage_pinned_targets_on_cpu

    def _sharded(self, local, shape):
        from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
            ShardedWeight,
        )

        return ShardedWeight("w", local, shape, shape, (0,) * len(shape))

    def test_contiguous_and_noncontiguous_targets(self):
        stage = self._stage()
        parent = _pinned([8, 6])
        ptr = parent.data_ptr()
        with _pinned_memory_as_cpu_backend():
            col_view = paddle.slice(parent, axes=[1], starts=[3], ends=[6])
        self.assertFalse(col_view.is_contiguous())
        whole = _pinned([4, 4])
        whole_ptr = whole.data_ptr()
        sd = {
            "view": self._sharded(col_view, (8, 3)),
            "whole": whole,
        }
        with stage(sd):
            # Inside the block every target is a CPU tensor.
            self.assertTrue(sd["view"].local_tensor.place.is_cpu_place())
            self.assertTrue(sd["whole"].place.is_cpu_place())
            sd["view"].local_tensor.copy_(paddle.full([8, 3], 5.0).cpu(), True)
            sd["whole"].copy_(paddle.full([4, 4], 9.0).cpu(), True)
        self.assertIs(sd["view"].local_tensor, col_view)
        self.assertIs(sd["whole"], whole)
        for t, p in ((parent, ptr), (whole, whole_ptr)):
            self.assertTrue(t.place.is_cuda_pinned_place(), str(t.place))
            self.assertEqual(t.data_ptr(), p)
        expected = np.arange(48, dtype="float32").reshape([8, 6])
        expected[:, 3:6] = 5.0
        np.testing.assert_array_equal(parent.numpy(), expected)
        np.testing.assert_array_equal(whole.numpy(), np.full([4, 4], 9.0))

    def test_restores_without_copy_on_error(self):
        stage = self._stage()
        t = _pinned([4])
        before = t.numpy().copy()
        sd = {"t": t}
        with self.assertRaises(RuntimeError), stage(sd):
            sd["t"].copy_(paddle.full([4], 3.0).cpu(), True)
            raise RuntimeError("load failed")
        self.assertIs(sd["t"], t)
        self.assertTrue(t.place.is_cuda_pinned_place())
        np.testing.assert_array_equal(t.numpy(), before)

    def test_noncontiguous_bfloat16_target(self):
        # A non-contiguous pinned view is staged through numpy, which returns
        # uint16 for bfloat16 and has to be viewed back to the original dtype.
        stage = self._stage()
        parent = paddle.zeros([8, 6], dtype="bfloat16").pin_memory()
        ptr = parent.data_ptr()
        with _pinned_memory_as_cpu_backend():
            view = paddle.slice(parent, axes=[1], starts=[2], ends=[5])
        self.assertFalse(view.is_contiguous())
        sd = {"view": self._sharded(view, (8, 3))}
        with stage(sd):
            staged = sd["view"].local_tensor
            self.assertTrue(staged.place.is_cpu_place())
            self.assertEqual(staged.dtype, paddle.bfloat16)
            staged.copy_(paddle.full([8, 3], 2.0, dtype="bfloat16").cpu(), True)
        self.assertTrue(parent.place.is_cuda_pinned_place())
        self.assertEqual(parent.data_ptr(), ptr)
        got = parent.astype("float32").numpy()
        np.testing.assert_allclose(got[:, 2:5], np.full([8, 3], 2.0), atol=1e-2)
        np.testing.assert_allclose(got[:, :2], np.zeros([8, 2]), atol=1e-2)

    def test_result_reshaped_back_to_target_shape(self):
        # The loader may hand back a flattened tensor (unflatten path); the
        # staged result is reshaped to the target shape before the copy back.
        stage = self._stage()
        parent = _pinned([4, 3])
        ptr = parent.data_ptr()
        sd = {"t": parent}
        with stage(sd):
            sd["t"] = paddle.arange(12, dtype="float32").cpu()
        self.assertIs(sd["t"], parent)
        self.assertTrue(parent.place.is_cuda_pinned_place())
        self.assertEqual(parent.data_ptr(), ptr)
        np.testing.assert_array_equal(
            parent.numpy(), np.arange(12, dtype="float32").reshape([4, 3])
        )

    def test_decorator_accepts_state_dict_keyword(self):
        from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
            _keep_pinned_targets,
        )

        seen = {}

        @_keep_pinned_targets
        def fake_load(state_dict, path):
            seen["place"] = state_dict["t"].place
            return "done"

        t = _pinned([4])
        self.assertEqual(fake_load(state_dict={"t": t}, path="p"), "done")
        self.assertTrue(seen["place"].is_cpu_place())
        self.assertTrue(t.place.is_cuda_pinned_place())

    def test_non_pinned_targets_untouched(self):
        stage = self._stage()
        gpu = paddle.zeros([4]).cuda()
        cpu = paddle.zeros([4]).cpu()
        sd = {"g": gpu, "c": cpu}
        with stage(sd):
            self.assertIs(sd["g"], gpu)
            self.assertIs(sd["c"], cpu)

    def test_decorator_keeps_public_signature(self):
        # The decorated load APIs are public, so their argument spec must stay
        # exactly as declared instead of degrading to (*args, **kwargs).
        from paddle.distributed.flex_checkpoint.dcp.load_state_dict import (
            _keep_pinned_targets,
        )

        def func(state_dict, path, offload: bool = False) -> None:
            pass

        spec = inspect.getfullargspec(_keep_pinned_targets(func))
        self.assertEqual(spec, inspect.getfullargspec(func))
        self.assertIsNone(spec.varargs)
        self.assertIsNone(spec.varkw)

        spec = inspect.getfullargspec(paddle.distributed.load_state_dict)
        self.assertEqual(spec.args[0], "state_dict")
        self.assertIsNone(spec.varargs)
        self.assertIsNone(spec.varkw)


if __name__ == '__main__':
    unittest.main()
