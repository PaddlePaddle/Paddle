# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_legacy_and_pt_and_pir,
)

import paddle
import paddle.inference as paddle_infer


class TestLayer1(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=True)

    def forward(self, x):
        for i in range(5):
            x = paddle.nn.functional.softmax(x, -1)
        x = x.cast("float32")
        x = self.func(x)
        return x

    def func(self, x):
        return self.fn(x)


class TestLayer2(paddle.nn.Layer):
    def __init__(self, hidd):
        super().__init__()
        self.fn = paddle.nn.Linear(hidd, hidd, bias_attr=True)
    def forward(self, x_list, bool_value, my_dict={}):
        x = x_list[0]
        y = my_dict["y"]
        y = paddle.nn.functional.relu(y)
        x = x + y
        for i in range(5):
            x = paddle.nn.functional.softmax(x, -1)
        x = x.cast("float32")
        x = self.fn(x)
        x = x + x_list[1]
        if bool_value:
            x = x * 3
        else:
            x = 2 * x
        return x

class TestToStaticInputListModel(Dy2StTestBase):
    @test_ast_only
    def test_dygraph_static_same_result(self):
        hidd = 1024
        batch = 4096
        dtype = "float32"
        x = paddle.rand([batch, hidd], dtype=dtype)
        my_layer = TestLayer2(hidd)
        my_dict = {"y": x+x}
        result0 = my_layer([x, x], bool_value=True, my_dict=my_dict).numpy()
        my_static_layer = paddle.incubate.jit.inference(my_layer)
        my_static_layer = paddle.incubate.jit.inference(my_layer)

        result1 = my_layer([x, x], bool_value=True, my_dict=my_dict).numpy()
        np.testing.assert_allclose(result0, result1, rtol=0.001, atol=1e-05)


if __name__ == '__main__':
    unittest.main()
