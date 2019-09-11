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

import os, shutil
import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.core import PaddleTensor
from paddle.fluid.core import PaddleDType


class TestInferenceApi(unittest.TestCase):
    def test_inference_api(self):
        tensor32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        paddletensor32 = PaddleTensor(tensor32)
        value32 = np.array(paddletensor32.data.int32_data()).reshape(*[20, 2])
        dtype32 = paddletensor32.dtype
        self.assertEqual(value32.all(), tensor32.all())
        self.assertEqual(dtype32, PaddleDType.INT32)
        self.assertEqual(
            type(paddletensor32.data.tolist('int32')), type(tensor32.tolist()))
        self.assertEqual(
            paddletensor32.data.tolist('int32'), tensor32.ravel().tolist())
        self.assertEqual(type(paddletensor32.as_ndarray()), type(tensor32))
        paddletensor32.data.reset(tensor32)
        self.assertEqual(paddletensor32.as_ndarray().all(), tensor32.all())

        tensor64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        paddletensor64 = PaddleTensor(tensor64)
        value64 = np.array(paddletensor64.data.int64_data()).reshape(*[20, 2])
        dtype64 = paddletensor64.dtype
        self.assertEqual(value64.all(), tensor64.all())
        self.assertEqual(dtype64, PaddleDType.INT64)
        self.assertEqual(
            type(paddletensor64.data.tolist('int64')), type(tensor64.tolist()))
        self.assertEqual(
            paddletensor64.data.tolist('int64'), tensor64.ravel().tolist())
        self.assertEqual(type(paddletensor64.as_ndarray()), type(tensor64))
        paddletensor64.data.reset(tensor64)
        self.assertEqual(paddletensor64.as_ndarray().all(), tensor64.all())

        tensor_float = np.random.randn(20, 2).astype('float32')
        paddletensor_float = PaddleTensor(tensor_float)
        value_float = np.array(paddletensor_float.data.float_data()).reshape(
            *[20, 2])
        dtype_float = paddletensor_float.dtype
        self.assertEqual(value_float.all(), tensor_float.all())
        self.assertEqual(dtype_float, PaddleDType.FLOAT32)
        self.assertEqual(
            type(paddletensor_float.data.tolist('float32')),
            type(tensor_float.tolist()))
        self.assertEqual(
            paddletensor_float.data.tolist('float32'),
            tensor_float.ravel().tolist())
        self.assertEqual(
            type(paddletensor_float.as_ndarray()), type(tensor_float))
        paddletensor_float.data.reset(tensor_float)
        self.assertEqual(paddletensor_float.as_ndarray().all(),
                         tensor_float.all())


if __name__ == '__main__':
    unittest.main()
