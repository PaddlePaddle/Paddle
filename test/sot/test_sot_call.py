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

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils import min_graph_size_guard


def simple_case(x, y):
    a = x[0]
    b = x[1]
    c = paddle.reshape(y, [a, b])
    # This case use full graph mode, we need to return a list to keep same as SOT
    return [c]


class TestSotCall(TestCaseBase):
    @min_graph_size_guard(0)
    def test_input_with_different_places(self):
        if paddle.device.is_compiled_with_cuda():
            a = paddle.to_tensor(
                [2, 3], dtype='int32', place=paddle.CUDAPlace(0)
            )
            b = paddle.ones([3, 2], dtype='int32')
            _, pp = paddle.jit.to_static(
                simple_case, full_graph=True
            ).get_concrete_program(a, b)
            c = paddle.to_tensor([2, 3], dtype='int32', place=paddle.CPUPlace())

            result1 = pp.sot_call([a, b])
            result2 = pp.sot_call([c, b])
            self.assert_nest_match(result1, result2)


if __name__ == "__main__":
    unittest.main()
