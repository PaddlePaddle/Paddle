#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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


class ApiOnesTest(unittest.TestCase):
    def test_paddle_ones(self):
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype="float32")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype="float64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype="float64")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=10, dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expected_result = np.ones(10, dtype="int64")
        self.assertEqual((result == expected_result).all(), True)


if __name__ == "__main__":
    unittest.main()
