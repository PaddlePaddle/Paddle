# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import get_device, get_places

import paddle
from paddle.base import core


class TensorFill_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [32, 32]

    def test_tensor_fill_true(self):
        typelist = ['float32', 'float64', 'int32', 'int64', 'float16']

        for idx, p in enumerate(get_places()):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            np_arr = np.reshape(
                np.array(range(np.prod(self.shape))), self.shape
            )
            for dtype in typelist:
                var = 1.0
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                target = tensor.numpy()
                target[...] = var

                tensor.fill_(var)  # var type is basic type in typelist
                self.assertEqual((tensor.numpy() == target).all(), True)

    def test_tensor_fill_backward(self):
        typelist = ['float32']

        for idx, p in enumerate(get_places()):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device(get_device())
            np_arr = np.reshape(
                np.array(range(np.prod(self.shape))), self.shape
            )
            for dtype in typelist:
                var = 1
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                tensor.stop_gradient = False
                y = tensor * 2
                y.retain_grads()
                y.fill_(var)
                loss = y.sum()
                loss.backward()

                self.assertEqual((y.grad.numpy() == 0).all().item(), True)

    def test_errors(self):
        def test_list():
            x = paddle.to_tensor([2, 3, 4])
            x.fill_([1])

        self.assertRaises(TypeError, test_list)

    def test_tensor_fill_pinned_memory(self):
        if not (
            core.is_compiled_with_cuda()
            or (
                hasattr(core, 'is_compiled_with_xpu')
                and core.is_compiled_with_xpu()
            )
        ):
            self.skipTest("Pinned memory requires CUDA or XPU backend")

        paddle.set_device('cpu')
        tensor = paddle.zeros([2], dtype='int32').pin_memory()
        self.assertTrue(
            tensor.place.is_cuda_pinned_place()
            or tensor.place.is_xpu_pinned_place()
        )

        tensor.fill_(123)
        self.assertTrue(
            tensor.place.is_cuda_pinned_place()
            or tensor.place.is_xpu_pinned_place()
        )
        np.testing.assert_array_equal(
            tensor.cpu().numpy(), np.array([123, 123], dtype='int32')
        )

        tensor.zero_()
        self.assertTrue(
            tensor.place.is_cuda_pinned_place()
            or tensor.place.is_xpu_pinned_place()
        )
        np.testing.assert_array_equal(
            tensor.cpu().numpy(), np.array([0, 0], dtype='int32')
        )


if __name__ == '__main__':
    unittest.main()
