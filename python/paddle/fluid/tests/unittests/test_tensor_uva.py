# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import unittest
import numpy as np


class TestTensorCopyFrom(unittest.TestCase):
    def test_main(self):
        if paddle.fluid.core.is_compiled_with_cuda():
            place = paddle.CPUPlace()
            np_value = np.random.random(size=[10, 30]).astype('float32')
            tensor = paddle.to_tensor(np_value, place=place)
            tensor._uva()
            self.assertTrue(tensor.place.is_gpu_place())


class TestUVATensorFromNumpy(unittest.TestCase):
    def test_uva_tensor_creation(self):
        if paddle.fluid.core.is_compiled_with_cuda():
            dtype_list = [
                "int32", "int64", "float32", "float64", "float16", "int8",
                "int16", "bool"
            ]
            for dtype in dtype_list:
                data = np.random.randint(10, size=[4, 5]).astype(dtype)
                tensor = paddle.fluid.core.to_uva_tensor(data, 0)
                self.assertTrue(tensor.place.is_gpu_place())
                self.assertTrue(np.allclose(tensor.numpy(), data))


if __name__ == "__main__":
    unittest.main()
