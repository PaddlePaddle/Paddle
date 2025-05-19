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

from test_case_base import TestCaseBase

import paddle
from paddle.jit import sot


class TestArrayWrite(TestCaseBase):
    def test_array_write(self):
        def call_array_api():
            arr = paddle.tensor.create_array(dtype="float32")
            x = paddle.full(shape=[1, 3], fill_value=5, dtype="float32")
            i = paddle.zeros(shape=[1], dtype="int32")
            # The `breakgraph` here is used to document that the `create_array` function has a dynamic-static inconsistency issue.
            # In SOT, it cannot be directly handled as a PaddleApiVariable; instead, it requires internal function simulation for execution.
            # Thus, a manual breakpoint is marked to flag this issue.
            sot.psdb.breakgraph()
            # The presence of the `item` method within `array_write` / `array_read` can cause breakgraph.
            arr = paddle.tensor.array_write(x, i, array=arr)
            item = paddle.tensor.array_read(arr, i)
            return item

        self.assert_results(call_array_api)


if __name__ == "__main__":
    unittest.main()
