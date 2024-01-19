#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle
from paddle.framework import get_default_dtype, set_default_dtype


class TestDefaultType(unittest.TestCase):
    def check_default(self):
        self.assertEqual("float32", get_default_dtype())

    def test_api(self):
        self.check_default()

        set_default_dtype("float64")
        self.assertEqual("float64", get_default_dtype())

        set_default_dtype("float32")
        self.assertEqual("float32", get_default_dtype())

        set_default_dtype("float16")
        self.assertEqual("float16", get_default_dtype())

        set_default_dtype("bfloat16")
        self.assertEqual("bfloat16", get_default_dtype())

        set_default_dtype(np.float64)
        self.assertEqual("float64", get_default_dtype())

        set_default_dtype(np.float32)
        self.assertEqual("float32", get_default_dtype())

        set_default_dtype(np.float16)
        self.assertEqual("float16", get_default_dtype())

        set_default_dtype(paddle.float64)
        self.assertEqual("float64", get_default_dtype())

        set_default_dtype(paddle.float32)
        self.assertEqual("float32", get_default_dtype())

        set_default_dtype(paddle.float16)
        self.assertEqual("float16", get_default_dtype())

        set_default_dtype(paddle.bfloat16)
        self.assertEqual("bfloat16", get_default_dtype())


class TestDefaultTypeInLayer(unittest.TestCase):
    def test_bfloat16(self):
        set_default_dtype("bfloat16")
        linear = paddle.nn.Linear(10, 20)
        self.assertEqual(linear.weight.dtype, paddle.bfloat16)


class TestRaiseError(unittest.TestCase):
    def test_error(self):
        self.assertRaises(TypeError, set_default_dtype, "int32")
        self.assertRaises(TypeError, set_default_dtype, np.int32)
        self.assertRaises(TypeError, set_default_dtype, paddle.int32)
        self.assertRaises(TypeError, set_default_dtype, "int64")
        self.assertRaises(TypeError, set_default_dtype, np.int64)
        self.assertRaises(TypeError, set_default_dtype, paddle.int64)


if __name__ == '__main__':
    unittest.main()
