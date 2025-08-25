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

import paddle


class TestClearParamStorage(unittest.TestCase):
    def test_clear_param_storage(self):
        class TestLayer(paddle.nn.Layer):
            def __init__(self, dtype):
                super().__init__()
                self._w = self.create_parameter([2, 3], dtype=dtype)
                self._b = self.create_parameter([2, 3], dtype=dtype)
                self._w.color = {"color": "_w"}
                self._b.color = {"color": "_b"}

            @paddle.amp.debugging.check_layer_numerics
            def forward(self, x):
                return x * self._w + self._b

        dtype = 'float32'
        model = TestLayer(dtype)
        adam = paddle.optimizer.Adam(parameters=model.parameters())
        adam.clear_param_storage("_w")
        adam.clear_param_storage("_b")
        adam.reset_param_storage()


if __name__ == '__main__':
    unittest.main()
