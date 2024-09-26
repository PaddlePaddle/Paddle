#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


def detach_fn(x, y):
    u = x + y
    v = u.detach()
    o1 = v + 1

    return o1, u


class TestDetach(Dy2StTestBase):
    def test_detach(self):
        static_fn = paddle.jit.to_static(detach_fn)
        x = paddle.ones([], 'float32')
        y = paddle.ones([], 'float32')
        static_res = static_fn(x, y)
        dygraph_res = detach_fn(x, y)
        np.testing.assert_allclose(
            static_res[0].numpy(), dygraph_res[0].numpy()
        )
        np.testing.assert_allclose(
            static_res[1].numpy(), dygraph_res[1].numpy()
        )


if __name__ == '__main__':
    unittest.main()
