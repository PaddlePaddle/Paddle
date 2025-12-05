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
from dygraph_to_static_utils import (
    Dy2StTestBase,
)

import paddle

SEED = 2025
np.random.seed(SEED)
paddle.seed(SEED)


def func(x, y):
    return x + y


class TestAddCastToElementwiseAddPass(Dy2StTestBase):
    # test AddCastToElementwiseAddPass
    def _run(self, dtype):
        static_fn = paddle.jit.to_static(func)
        x = paddle.randn([200, 200])
        y = paddle.randn([200, 200], dtype=dtype)
        np.testing.assert_allclose(
            static_fn(x, y).numpy(), x.numpy() + y.cast("float32").numpy()
        )

    def test_bf16(self):
        self._run(dtype="bfloat16")

    def test_fp16(self):
        self._run(dtype="float16")


if __name__ == '__main__':
    unittest.main()
