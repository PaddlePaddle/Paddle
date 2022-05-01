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

import paddle.fluid as fluid
import unittest
import numpy as np
import six
import paddle
from paddle.fluid.framework import _test_eager_guard


class TensorFill_Test(unittest.TestCase):
    def setUp(self):
        self.shape = [32, 32]

    def func_test_tensor_fill_true(self):
        typelist = ['float32', 'float64', 'int32', 'int64', 'float16']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
            places.append(fluid.CUDAPinnedPlace())

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            np_arr = np.reshape(
                np.array(six.moves.range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                var = 1.
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                target = tensor.numpy()
                target[...] = var

                tensor.fill_(var)  #var type is basic type in typelist
                self.assertEqual((tensor.numpy() == target).all(), True)

    def test_tensor_fill_true(self):
        with _test_eager_guard():
            self.func_test_tensor_fill_true()
        self.func_test_tensor_fill_true()

    def func_test_tensor_fill_backward(self):
        typelist = ['float32']
        places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
            places.append(fluid.CUDAPinnedPlace())

        for idx, p in enumerate(places):
            if idx == 0:
                paddle.set_device('cpu')
            else:
                paddle.set_device('gpu')
            np_arr = np.reshape(
                np.array(six.moves.range(np.prod(self.shape))), self.shape)
            for dtype in typelist:
                var = int(1)
                tensor = paddle.to_tensor(np_arr, place=p, dtype=dtype)
                tensor.stop_gradient = False
                y = tensor * 2
                y.fill_(var)
                loss = y.sum()
                loss.backward()

                self.assertEqual((y.grad.numpy() == 0).all().item(), True)

    def test_tensor_fill_backward(self):
        with _test_eager_guard():
            self.func_test_tensor_fill_backward()
        self.func_test_tensor_fill_backward()

    def func_test_errors(self):
        def test_list():
            x = paddle.to_tensor([2, 3, 4])
            x.fill_([1])

        self.assertRaises(TypeError, test_list)

    def test_errors(self):
        with _test_eager_guard():
            self.func_test_errors()
        self.func_test_errors()


if __name__ == '__main__':
    unittest.main()
