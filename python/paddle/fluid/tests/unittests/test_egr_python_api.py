# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
import paddle.fluid.eager.eager_tensor_patch_methods as eager_tensor_patch_methods
import paddle
import numpy as np
from paddle.fluid import eager_guard
import unittest


class EagerScaleTestCase(unittest.TestCase):
    def test_scale_base(self):
        with eager_guard():
            paddle.set_device("cpu")
            arr = np.ones([4, 16, 16, 32]).astype('float32')
            tensor = paddle.to_tensor(arr, 'float32', core.CPUPlace())
            print(tensor)
            tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            for i in range(0, 100):
                tensor = core.eager.scale(tensor, 2.0, 0.9, True, False)
            print(tensor)
            self.assertEqual(tensor.shape, [4, 16, 16, 32])
            self.assertEqual(tensor.stop_gradient, True)
            tensor.stop_gradient = False
            self.assertEqual(tensor.stop_gradient, False)
            tensor.stop_gradient = True
            self.assertEqual(tensor.stop_gradient, False)
            tensor.name = 'tensor_name_test'
            self.assertEqual(tensor.name, 'tensor_name_test')
            self.assertEqual(tensor.persistable, False)
            tensor.persistable = True
            self.assertEqual(tensor.persistable, True)
            tensor.persistable = False
            self.assertEqual(tensor.persistable, False)
            self.assertTrue(tensor.place.is_cpu_place())


class EagerDtypeTestCase(unittest.TestCase):
    def check_to_tesnsor_and_numpy(self, dtype):
        with eager_guard():
            arr = np.random.random([4, 16, 16, 32]).astype(dtype)
            tensor = paddle.to_tensor(arr, dtype)
            self.assertEqual(tensor.dtype, dtype)
            self.assertTrue(np.array_equal(arr, tensor.numpy()))

    def test_dtype_base(self):
        self.check_to_tesnsor_and_numpy('bool')
        self.check_to_tesnsor_and_numpy('int8')
        self.check_to_tesnsor_and_numpy('uint8')
        self.check_to_tesnsor_and_numpy('int16')
        self.check_to_tesnsor_and_numpy('int32')
        self.check_to_tesnsor_and_numpy('int64')
        self.check_to_tesnsor_and_numpy('float16')
        self.check_to_tesnsor_and_numpy('float32')
        self.check_to_tesnsor_and_numpy('float64')
        self.check_to_tesnsor_and_numpy('complex64')
        self.check_to_tesnsor_and_numpy('complex128')


if __name__ == "__main__":
    unittest.main()
