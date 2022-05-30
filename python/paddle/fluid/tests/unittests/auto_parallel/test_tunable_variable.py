# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distributed.auto_parallel.tuner import tunable_variable as tv


class TestTunableVariable(unittest.TestCase):
    def test_fixed(self):
        fixed = tv.Fixed("fixed", True)
        fixed = tv.Fixed.from_state(fixed.get_state())
        self.assertEqual(fixed.default, True)
        self.assertEqual(fixed.random(), True)

        fixed = tv.Fixed("fixed", 1)
        fixed = tv.Fixed.from_state(fixed.get_state())
        self.assertEqual(fixed.default, 1)
        self.assertEqual(fixed.random(), 1)

    def test_boolean(self):
        boolean = tv.Boolean("bool")
        boolean = tv.Boolean.from_state(boolean.get_state())
        self.assertEqual(boolean.default, False)
        self.assertIn(boolean.random(), [True, False])
        self.assertIn(boolean.random(1234), [True, False])

        boolean = tv.Boolean("bool", True)
        boolean = tv.Boolean.from_state(boolean.get_state())
        self.assertEqual(boolean.default, True)
        self.assertIn(boolean.random(), [True, False])
        self.assertIn(boolean.random(1234), [True, False])

    def test_choice(self):
        choice = tv.Choice("choice", [1, 2, 3, 4])
        choice = tv.Choice.from_state(choice.get_state())
        self.assertEqual(choice.default, 1)
        self.assertIn(choice.random(), [1, 2, 3, 4])
        self.assertIn(choice.random(1234), [1, 2, 3, 4])

        choice = tv.Choice("choice", [1, 2, 3, 4], default=2)
        choice = tv.Choice.from_state(choice.get_state())
        self.assertEqual(choice.default, 2)
        self.assertIn(choice.random(), [1, 2, 3, 4])
        self.assertIn(choice.random(1234), [1, 2, 3, 4])

    def test_int_range(self):
        int_range = tv.IntRange("int_range", start=1, stop=4, default=2)
        int_range = tv.IntRange.from_state(int_range.get_state())
        self.assertEqual(int_range.default, 2)
        self.assertIn(int_range.random(), [1, 2, 3, 4])
        self.assertIn(int_range.random(1234), [1, 2, 3, 4])
        self.assertNotEqual(int_range.default, 4)

        int_range = tv.IntRange(
            "int_range", start=1, stop=8, step=2, default=3, endpoint=True)
        int_range = tv.IntRange.from_state(int_range.get_state())
        self.assertEqual(int_range.default, 3)
        self.assertIn(int_range.random(), [1, 3, 5, 7])
        self.assertIn(int_range.random(1234), [1, 3, 5, 7])
        self.assertNotEqual(int_range.default, 2)

    def test_float_range(self):
        float_range = tv.FloatRange(
            "float_range", start=0.4, stop=4.4, default=2.0)
        float_range = tv.FloatRange.from_state(float_range.get_state())
        self.assertEqual(float_range.default, 2.0)
        self.assertGreaterEqual(float_range.random(), 0.4)
        self.assertLess(float_range.random(1234), 4.4)
        self.assertNotAlmostEqual(float_range.random(), 1)
        self.assertNotAlmostEqual(float_range.random(), 4.4)

        float_range = tv.FloatRange(
            "float_range",
            start=0.4,
            stop=8.4,
            step=2.0,
            default=3.0,
            endpoint=True)
        float_range = tv.FloatRange.from_state(float_range.get_state())
        self.assertEqual(float_range.default, 3.0)
        self.assertGreaterEqual(float_range.random(), 0.4)
        self.assertLessEqual(float_range.random(1234), 8.4)
        self.assertNotAlmostEqual(float_range.random(), 2)


if __name__ == "__main__":
    unittest.main()
