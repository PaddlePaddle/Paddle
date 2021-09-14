# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest

import numpy as np
import paddle
import unittest


class TestCheckFetchList(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.feed = {"x": np.array([[0], [0], [1], [0]], dtype='float32')}
        self.expected = np.array([[0], [1], [0]], dtype='float32')

    def build_program(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
            output = paddle.unique_consecutive(
                x, return_inverse=True, return_counts=True, axis=0)

        return main_program, output

    def test_raise_error(self):
        main_program, fetch_list = self.build_program()

        exe = paddle.static.Executor(paddle.CPUPlace())
        with self.assertRaises(TypeError):
            res = exe.run(
                main_program,
                feed=self.feed,
                fetch_list=[fetch_list],  # not support nested list/tuple
                return_numpy=True)

    def test_fetch(self):
        main_program, fetch_list = self.build_program()

        exe = paddle.static.Executor(paddle.CPUPlace())
        res = exe.run(main_program,
                      feed=self.feed,
                      fetch_list=[fetch_list[0]],
                      return_numpy=True)

        self.assertTrue(np.array_equal(res[0], self.expected))


if __name__ == '__main__':
    unittest.main()
