# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import paddle


class TestFromNumpy(unittest.TestCase):
    def setUp(self):
        self.shape = [3, 4, 5]
        self.dtypes = [
            "bool",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "complex64",
            "complex128",
        ]
        self.devices = ["cpu", paddle.CPUPlace()]
        if paddle.base.is_compiled_with_cuda():
            self.devices.append("gpu")
            self.devices.append(paddle.CUDAPlace(0))
        self.stop_gradients = [True, False]

    def prepare_data(self, dtype):
        if dtype == "bool":
            return np.random.randint(0, 2, self.shape)
        else:
            return np.random.randn(*self.shape)

    def test_base(self):
        for dtype in self.dtypes:
            np_data = self.prepare_data(dtype)
            for device in self.devices:
                for stop_gradient in self.stop_gradients:
                    tensor = paddle.asarray(
                        np_data,
                        device=device,
                        requires_grad=stop_gradient,
                        dtype=dtype,
                    )
                    target_place = device
                    if isinstance(target_place, str):
                        target_place = (
                            paddle.CPUPlace()
                            if target_place == "cpu"
                            else paddle.CUDAPlace(0)
                        )
                    self.assertEqual(tensor.stop_gradient, not stop_gradient)
                    self.assertEqual(tensor.place, target_place)
                    np.testing.assert_allclose(
                        tensor.numpy(), np_data.astype(dtype)
                    )


if __name__ == "__main__":
    unittest.main()
