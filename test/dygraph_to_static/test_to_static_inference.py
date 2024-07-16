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


class TestLayer(paddle.nn.Layer):
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


class TestToStaticInfenrenceModel(Dy2StTestBase):
    @test_ast_only
    @test_legacy_and_pt_and_pir
    def test_dygraph_static_same_result(self):
        hidd = 1024
        batch = 4096
        hidd = 1024
        dtype = "float32"
        x = paddle.rand([batch, hidd], dtype=dtype)
        my_layer = TestLayer(hidd)
        result0 = my_layer(x).numpy()
        my_static_layer = paddle.jit.to_static(my_layer, backend="inference")
        result1 = my_layer(x).numpy()
        np.testing.assert_allclose(result0, result1, rtol=0.001, atol=1e-05)


class TestToStaticInfenrenceFunc(Dy2StTestBase):
    @test_ast_only
    @test_legacy_and_pt_and_pir
    def test_dygraph_static_same_result(self):
        hidd = 1024
        batch = 4096
        hidd = 1024
        dtype = "float32"
        # test dynamic shape
        x = paddle.rand([batch, hidd], dtype=dtype)
        y = paddle.rand([batch + 1, hidd], dtype=dtype)
        my_layer = TestLayer(hidd)
        result_x0 = my_layer(x).numpy()
        result_y0 = my_layer(y).numpy()

        my_layer.func = paddle.jit.to_static(my_layer.func, backend="inference")

        result_x1 = my_layer(x).numpy()
        result_y1 = my_layer(y).numpy()
        np.testing.assert_allclose(result_x0, result_x1, rtol=0.001, atol=1e-05)
        np.testing.assert_allclose(result_y0, result_y1, rtol=0.001, atol=1e-05)


if __name__ == '__main__':
    unittest.main()
