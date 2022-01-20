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
import paddle
paddle.enable_static()
import numpy as np
from paddle.fluid.core import DistModelTensor
from paddle.fluid.core import DistModelDataType


class TestInferenceApi(unittest.TestCase):
    def test_inference_api(self):
        tensor32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        dist_tensor32 = DistModelTensor(tensor32, '32_tensor')
        dtype32 = dist_tensor32.dtype
        self.assertEqual(dtype32, DistModelDataType.INT32)
        self.assertEqual(
            dist_tensor32.data.tolist('int32'), tensor32.ravel().tolist())
        self.assertEqual(dist_tensor32.data.length(), 40)
        self.assertEqual(dist_tensor32.name, '32_tensor')
        dist_tensor32.data.reset(tensor32)
        self.assertEqual(dist_tensor32.as_ndarray().ravel().tolist(),
                         tensor32.ravel().tolist())

        tensor64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        dist_tensor64 = DistModelTensor(tensor64, '64_tensor')
        dtype64 = dist_tensor64.dtype
        self.assertEqual(dtype64, DistModelDataType.INT64)
        self.assertEqual(
            dist_tensor64.data.tolist('int64'), tensor64.ravel().tolist())
        self.assertEqual(dist_tensor64.data.length(), 40)
        self.assertEqual(dist_tensor64.name, '64_tensor')
        dist_tensor64.data.reset(tensor64)
        self.assertEqual(dist_tensor64.as_ndarray().ravel().tolist(),
                         tensor32.ravel().tolist())

        tensor_float = np.random.randn(20, 2).astype('float32')
        dist_tensor_float = DistModelTensor(tensor_float, 'float_tensor')
        dtype_float = dist_tensor_float.dtype
        self.assertEqual(dtype_float, DistModelDataType.FLOAT32)
        self.assertEqual(
            dist_tensor_float.data.tolist('float32'),
            tensor_float.ravel().tolist())
        self.assertEqual(dist_tensor_float.data.length(), 40)
        self.assertEqual(dist_tensor_float.name, 'float_tensor')
        dist_tensor_float.data.reset(tensor_float)
        self.assertEqual(dist_tensor_float.as_ndarray().ravel().tolist(),
                         tensor32.ravel().tolist())


if __name__ == '__main__':
    unittest.main()
