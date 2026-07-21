# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from unittest import mock

import paddle
from paddle.distributed.fleet.utils import tensor_fusion_helper
from paddle.distributed.fleet.utils.tensor_fusion_helper import (
    HOOK_ACTION,
    FusedCommBuffer,
)


class TestFusedCommBufferGradChecker(unittest.TestCase):
    def test_fused_comm_buffer_grad_checker(self):
        linear = paddle.nn.Linear(10, 10)
        w = linear.weight
        b = linear.bias
        w.main_grad = None
        b.main_grad = None
        buffer = FusedCommBuffer(
            id=0,
            params=[w, b],
            comm_group=None,
            acc_steps=10,
            act=HOOK_ACTION.ALL_REDUCE,
        )
        assert buffer.use_main_grad
        buffer.add_grad(w)
        buffer.add_grad(b)
        w.main_grad = paddle.to_tensor([1], stop_gradient=True, dtype="float32")
        try:
            buffer.add_grad(w)
            raise AssertionError(
                "Above add_grad should raise value error, this assertion should be unreachable."
            )
        except ValueError:
            pass


class _FakeTensor:
    def __init__(self, ptr=1):
        self._ptr = ptr

    def data_ptr(self):
        return self._ptr

    def _share_buffer_to(self, other):
        self.shared_to = other

    def _clear_to_zero_allocation(self):
        self.cleared_to_zero = True


class _FakeParam:
    name = "fake_param"


class _FakeParamView:
    def __init__(self):
        self.clear_param_buffer_count = 0
        self.reset_param_buffer_arg = None

    def _clear_param_buffer(self):
        self.clear_param_buffer_count += 1

    def _reset_param_buffer(self, tensor):
        self.reset_param_buffer_arg = tensor


class TestFusedCommBufferIPCMeta(unittest.TestCase):
    def test_param_buffer_ipc_meta_cache_tracks_data_ptr(self):
        buffer = FusedCommBuffer.__new__(FusedCommBuffer)
        buffer._param_buffer_meta_tensor = _FakeTensor(ptr=10)
        buffer._param_buffer_ipc_meta = None
        buffer._param_buffer_ipc_meta_ptr = None

        with mock.patch.object(
            tensor_fusion_helper,
            "_share_tensor_ipc_meta",
            side_effect=["meta-10", "meta-20"],
        ) as share:
            self.assertEqual(buffer.param_buffer_ipc_meta, "meta-10")
            self.assertEqual(buffer.param_buffer_ipc_meta, "meta-10")
            self.assertEqual(share.call_count, 1)

            buffer._param_buffer_meta_tensor = _FakeTensor(ptr=20)
            self.assertEqual(buffer.param_buffer_ipc_meta, "meta-20")
            self.assertEqual(share.call_count, 2)

        buffer._param_buffer_meta_tensor = None
        self.assertIsNone(buffer.param_buffer_ipc_meta)

    def test_param_storage_clear_and_reset_clear_ipc_meta_cache(self):
        buffer = FusedCommBuffer.__new__(FusedCommBuffer)
        param = _FakeParam()
        view = _FakeParamView()
        buffer._params = [param]
        buffer._sharding_param_grad_view = {param.name: view}
        buffer._param_buffer_ipc_meta = "old-meta"
        buffer._param_buffer_ipc_meta_ptr = 123
        buffer.param_storage = _FakeTensor(ptr=1)

        buffer._clear_param_storage()
        self.assertIsNone(buffer._param_buffer_ipc_meta)
        self.assertIsNone(buffer._param_buffer_ipc_meta_ptr)
        self.assertTrue(buffer.param_storage.cleared_to_zero)
        self.assertEqual(view.clear_param_buffer_count, 1)

        buffer._param_buffer_ipc_meta = "old-meta"
        buffer._param_buffer_ipc_meta_ptr = 123
        with mock.patch.object(
            tensor_fusion_helper.paddle,
            "empty_like",
            return_value=_FakeTensor(ptr=2),
        ):
            buffer._reset_param_storage()

        self.assertIsNone(buffer._param_buffer_ipc_meta)
        self.assertIsNone(buffer._param_buffer_ipc_meta_ptr)
        self.assertIsNotNone(view.reset_param_buffer_arg)


if __name__ == "__main__":
    unittest.main()
