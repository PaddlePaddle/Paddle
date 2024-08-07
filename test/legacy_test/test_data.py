# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.base import Program, program_guard


class TestApiStaticDataError(unittest.TestCase):
    def test_dtype(self):
        with program_guard(Program(), Program()):
            x1 = paddle.static.data(name="x1", shape=[2, 25])
            self.assertEqual(x1.dtype, paddle.float32)

            x2 = paddle.static.data(name="x2", shape=[2, 25], dtype="bool")
            self.assertEqual(x2.dtype, paddle.bool)

            paddle.set_default_dtype("float64")
            x3 = paddle.static.data(name="x3", shape=[2, 25])
            self.assertEqual(x3.dtype, paddle.float64)

    def test_0D(self):
        with program_guard(Program(), Program()):
            x1 = paddle.static.data(name="x1_0D", shape=[])
            self.assertEqual(x1.dtype, paddle.float32)
            x2 = paddle.static.data(name="x2_0D", shape=(), dtype="bool")
            self.assertEqual(x2.dtype, paddle.bool)

    def test_error(self):
        with program_guard(Program(), Program()):
            # 1. The type of 'name' in paddle.static.data must be str.
            def test_name_type():
                paddle.static.data(name=1, shape=[2, 25], dtype="bool")

            self.assertRaises(TypeError, test_name_type)

            # 2. The type of 'shape' in paddle.static.data must be list or tuple.
            def test_shape_type():
                paddle.static.data(name='data1', shape=2, dtype="bool")

            self.assertRaises(TypeError, test_shape_type)

        with program_guard(Program(), Program()):
            # 1. The type of 'name' in paddle.static.data must be str.
            def test_name_type():
                paddle.static.data(name=1, shape=[2, 25], dtype="bool")

            self.assertRaises(TypeError, test_name_type)

            # 2. The type of 'shape' in paddle.static.data must be list or tuple.
            def test_shape_type():
                paddle.static.data(name='data1', shape=2, dtype="bool")

            self.assertRaises(TypeError, test_shape_type)


class TestApiErrorWithDynamicMode(unittest.TestCase):
    def test_error(self):
        with program_guard(Program(), Program()):
            paddle.disable_static()
            self.assertRaises(AssertionError, paddle.static.data, 'a', [2, 25])
            paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
