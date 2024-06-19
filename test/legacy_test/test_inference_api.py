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

from paddle.base.core import PaddleDType, PaddleTensor


class TestInferenceApi(unittest.TestCase):
    def test_inference_api(self):
        tensor32 = np.random.randint(10, 20, size=[20, 2]).astype('int32')
        paddletensor32 = PaddleTensor(tensor32)
        dtype32 = paddletensor32.dtype
        self.assertEqual(dtype32, PaddleDType.INT32)
        self.assertEqual(
            paddletensor32.data.tolist('int32'), tensor32.ravel().tolist()
        )
        paddletensor32.data.reset(tensor32)
        self.assertEqual(
            paddletensor32.as_ndarray().ravel().tolist(),
            tensor32.ravel().tolist(),
        )

        tensor64 = np.random.randint(10, 20, size=[20, 2]).astype('int64')
        paddletensor64 = PaddleTensor(tensor64)
        dtype64 = paddletensor64.dtype
        self.assertEqual(dtype64, PaddleDType.INT64)
        self.assertEqual(
            paddletensor64.data.tolist('int64'), tensor64.ravel().tolist()
        )
        paddletensor64.data.reset(tensor64)
        self.assertEqual(
            paddletensor64.as_ndarray().ravel().tolist(),
            tensor64.ravel().tolist(),
        )

        tensor_float = np.random.randn(20, 2).astype('float32')
        paddletensor_float = PaddleTensor(tensor_float)
        dtype_float = paddletensor_float.dtype
        self.assertEqual(dtype_float, PaddleDType.FLOAT32)
        self.assertEqual(
            paddletensor_float.data.tolist('float32'),
            tensor_float.ravel().tolist(),
        )
        paddletensor_float.data.reset(tensor_float)
        self.assertEqual(
            paddletensor_float.as_ndarray().ravel().tolist(),
            tensor_float.ravel().tolist(),
        )


if __name__ == '__main__':
    unittest.main()
