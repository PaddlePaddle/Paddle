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
    def test_static_ones(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(10, dtype=paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(10, 2, 3, dtype=paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones([10, 2, 3], dtype=paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(size=[10, 2, 3], dtype=paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones([10, 2, 3], paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones([10, 2, 3], paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10, 2, 3], dtype=paddle.float32)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10])
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones(10, dtype="float32")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype="float64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones(10, dtype="float64")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=[10], dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones(10, dtype="int64")
        np.testing.assert_equal(result, expect)

        with paddle.static.program_guard(paddle.static.Program()):
            ones = paddle.ones(shape=10, dtype="int64")
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            (result,) = exe.run(fetch_list=[ones])
            expect = np.ones(10, dtype="int64")
        np.testing.assert_equal(result, expect)
        paddle.disable_static()

    def test_dygraph_ones(self):
        paddle.disable_static()
        result = paddle.ones(10, dtype=paddle.float32)
        expect = np.ones([10], dtype="float32")
        np.testing.assert_equal(result, expect)

        result = paddle.ones(10, 2, 3, dtype=paddle.float32)
        expect = np.ones([10, 2, 3], dtype="float32")
        np.testing.assert_equal(result, expect)

        result = paddle.ones([10, 2, 3], dtype=paddle.float32)
        np.testing.assert_equal(result, expect)

        result = paddle.ones(size=[10, 2, 3], dtype=paddle.float32)
        np.testing.assert_equal(result, expect)

        result = paddle.ones([10, 2, 3], paddle.float32)
        np.testing.assert_equal(result, expect)

        result = paddle.ones([10, 2, 3], "float32")
        np.testing.assert_equal(result, expect)

        result = paddle.ones(shape=[10, 2, 3], dtype=paddle.float32)
        np.testing.assert_equal(result, expect)


if __name__ == "__main__":
    unittest.main()
