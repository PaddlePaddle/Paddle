#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
    test_legacy_and_pt,
)

import paddle

np.random.seed(1)


def func(x):
    y = x[0:3].astype("float32")
    return y


class TestAmp64Case(Dy2StTestBase):
    def _run_static(self):
        static_func = paddle.jit.to_static(func)
        x = paddle.randn((10, 10)).astype("float64")
        with paddle.amp.auto_cast(True, level="O2"):
            dy_out = func(x)
            st_out = static_func(x)
        np.testing.assert_allclose(dy_out.numpy(), st_out.numpy())

    @test_legacy_and_pt
    def test_ast_to_func(self):
        self._run_static()


if __name__ == '__main__':
    unittest.main()
