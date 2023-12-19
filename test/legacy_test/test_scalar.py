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
import op

from paddle.base import framework


class TestWarpAsScalar(unittest.TestCase):
    def test_for_int(self):
        s = framework.wrap_as_scalar(np.iinfo(np.int64).max)
        self.assertEqual(s, np.iinfo(np.int64).max)

    def test_for_float(self):
        maximum = float(np.finfo(np.float64).max)
        s = framework.wrap_as_scalar(maximum)
        self.assertEqual(s, maximum)

    def test_for_bool(self):
        s = framework.wrap_as_scalar(True)
        self.assertEqual(s, True)

    def test_for_complex(self):
        c = 42.1 + 42.1j
        s = framework.wrap_as_scalar(c)
        self.assertEqual(s, c)

    def test_for_numpy_scalar(self):
        maximum = np.finfo(np.float64).max
        s = framework.wrap_as_scalar(maximum)
        self.assertEqual(s, maximum)

    def test_for_scalar(self):
        s1 = framework.wrap_as_scalar(42)
        s2 = framework.wrap_as_scalar(s1)
        self.assertEqual(s2, s1)

    def test_for_exception(self):
        with self.assertRaises(TypeError):
            framework.wrap_as_scalar("abc")


class TestWarpAsScalars(unittest.TestCase):
    def test_rewrap(self):
        vec = [framework.wrap_as_scalar(item) for item in (1, 2, 3, 4)]
        vec2 = framework.wrap_as_scalars(vec)
        self.assertListEqual(vec, vec2)

    def test_numpy_array(self):
        arr = np.random.randn(2, 3).astype(np.float64)
        scalars = framework.wrap_as_scalars(arr)
        values = framework.extract_plain_list(scalars)
        self.assertListEqual(arr.ravel().tolist(), values)

    def test_numeric_list(self):
        arr = [1 + 2j, 3 + 4j]
        scalars = framework.wrap_as_scalars(arr)
        values = framework.extract_plain_list(scalars)
        self.assertListEqual(arr, values)


class TestScalarValue(unittest.TestCase):
    def test_for_int(self):
        s = framework.wrap_as_scalar(np.iinfo(np.int64).max)
        self.assertEqual(s.value(), np.iinfo(np.int64).max)

    def test_for_float(self):
        maximum = float(np.finfo(np.float64).max)
        s = framework.wrap_as_scalar(maximum)
        self.assertEqual(s.value(), maximum)

    def test_for_bool(self):
        s = framework.wrap_as_scalar(True)
        self.assertEqual(s.value(), True)

    def test_for_complex(self):
        c = 42.1 + 42.1j
        s = framework.wrap_as_scalar(c)
        self.assertEqual(s.value(), c)

    def test_for_numpy_scalar(self):
        maximum = np.finfo(np.float64).max
        s = framework.wrap_as_scalar(maximum)
        self.assertEqual(s.value(), float(maximum))

    def test_for_scalar(self):
        s1 = framework.wrap_as_scalar(42)
        s2 = framework.wrap_as_scalar(s1)
        self.assertEqual(s2.value(), s1.value())


class TestScalarProto(unittest.TestCase):
    def test_make_scalar_proto_for_int(self):
        s = op.make_scalar_proto(42)
        self.assertEqual(s.i, 42)

    def test_make_scalar_proto_for_float(self):
        s = op.make_scalar_proto(42.1)
        self.assertEqual(s.r, 42.1)

    def test_make_scalar_proto_for_bool(self):
        s = op.make_scalar_proto(True)
        self.assertEqual(s.b, True)

    def test_make_scalar_proto_for_complex(self):
        s = op.make_scalar_proto(42.1 + 42.2j)
        self.assertEqual(s.c.r, 42.1)
        self.assertEqual(s.c.i, 42.2)
