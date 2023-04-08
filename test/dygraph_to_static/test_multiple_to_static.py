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

import numpy as np

import paddle
from paddle.jit import to_static


@paddle.jit.not_to_static
def part1(x):
    return x + 1


@paddle.jit.not_to_static
def part2(x):
    return x + x


def foo(x):
    x = to_static(part1)(x)

    # It will enter a new unique_name guard, so before applying this fix,
    # the name of x will be conflict with the name of x in part2 (they are
    # both `tmp_0`)
    paddle.enable_static()
    paddle.disable_static()

    x = to_static(part2)(x)
    return x


class TestMultipleToStaticNameConflict(unittest.TestCase):
    def test_multiple_to_static(self):
        x = paddle.to_tensor([4.0])

        paddle.jit.enable_to_static(False)
        out_dygraph = foo(x)
        paddle.jit.enable_to_static(True)
        out_static = foo(x)
        np.testing.assert_allclose(out_dygraph.numpy(), out_static.numpy())


if __name__ == "__main__":
    unittest.main()
