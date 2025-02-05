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

import paddle

paddle.enable_static()


class TestEliminateTransposePass(unittest.TestCase):
    def test_to_static_program(self):
        a = paddle.rand([1, 1024, 4096])
        b = paddle.reshape(a, [1024, 1, 4096])
        c = paddle.transpose(a, [1, 0, 2])

        paddle.seed(2024)
        program = paddle.static.default_main_program()
        exe = paddle.static.Executor()
        (
            b1,
            c1,
        ) = exe.run(program, fetch_list=[b, c])
        np.testing.assert_equal(b1, c1)

        paddle.seed(2024)
        program._pass_opt = {'pass_list': ['eliminate_transpose']}
        (c2,) = exe.run(program, fetch_list=[c])
        np.testing.assert_equal(c1, c2)


if __name__ == "__main__":
    unittest.main()
