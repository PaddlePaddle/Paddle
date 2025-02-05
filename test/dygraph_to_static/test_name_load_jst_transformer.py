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

import numpy
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
)

import paddle


def g(x):
    return x


def f(x):
    x[0] = 0
    y = g(x.reshape([1, 3])).reshape([3, 1])
    return y


class TestAssertVariable(Dy2StTestBase):
    @test_ast_only
    def test_load_name_in_args(self):
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        dy_out = f(x)
        x = paddle.to_tensor([1, 2, 3], dtype='float32')
        st_out = paddle.jit.to_static(f, full_graph=True)(x)
        numpy.testing.assert_allclose(dy_out, st_out)


if __name__ == '__main__':
    unittest.main()
