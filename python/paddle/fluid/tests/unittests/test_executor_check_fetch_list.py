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

import unittest

import numpy as np
import paddle
import unittest


class TestCheckFetchList(unittest.TestCase):

    def setUp(self):
        paddle.enable_static()
        self.feed = {"x": np.array([[0], [0], [1], [0]], dtype='float32')}
        self.expected = np.array([[0], [1], [0]], dtype='float32')
        self.build_program()
        self.exe = paddle.static.Executor(paddle.CPUPlace())

    def build_program(self):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            x = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
            output = paddle.unique_consecutive(x,
                                               return_inverse=True,
                                               return_counts=True,
                                               axis=0)

        self.main_program = main_program
        self.fetch_list = output

    def test_with_tuple(self):

        res = self.exe.run(
            self.main_program,
            feed=self.feed,
            fetch_list=[self.fetch_list],  # support single list/tuple
            return_numpy=True)

        np.testing.assert_array_equal(res[0], self.expected)

    def test_with_error(self):
        with self.assertRaises(TypeError):
            fetch_list = [23]
            res = self.exe.run(self.main_program,
                               feed=self.feed,
                               fetch_list=fetch_list)

        with self.assertRaises(TypeError):
            fetch_list = [(self.fetch_list[0], 32)]
            res = self.exe.run(self.main_program,
                               feed=self.feed,
                               fetch_list=fetch_list)


if __name__ == '__main__':
    unittest.main()
