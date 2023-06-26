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

paddle.enable_static()


class TestNewIr(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        x = paddle.ones([2, 2], dtype="float32")
        y = paddle.ones([2, 2], dtype="float32")

        z = x + y
        out = exe.run(
            paddle.static.default_main_program(), {}, fetch_list=[z.name]
        )

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


class TestCombineOp(unittest.TestCase):
    def test_with_new_ir(self):
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)

        x = paddle.ones([2, 2], dtype="float32")
        y = paddle.ones([2, 2], dtype="float32")

        z = paddle.linalg.multi_dot([x, y])
        out = exe.run(
            paddle.static.default_main_program(), {}, fetch_list=[z.name]
        )

        gold_res = np.ones([2, 2], dtype="float32") * 2

        np.testing.assert_array_equal(out[0], gold_res)


if __name__ == "__main__":
    unittest.main()
