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

from paddle.distributed.auto_parallel.static.tuner import tunable_space as ts


class TestTunableSpace(unittest.TestCase):
    def test_fixed(self):
        space = ts.TunableSpace()
        fixed = space.fixed("fixed", default=4)
        self.assertEqual(space.values["fixed"], 4)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["fixed"].name, "fixed")

        space.values["fixed"] = 2
        self.assertEqual(space.get_value("fixed"), 2)
        self.assertEqual(space.values, {"fixed": 2})
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["fixed"].name, "fixed")

    def test_boolean(self):
        space = ts.TunableSpace()
        boolean = space.boolean("boolean")
        self.assertEqual(space.values["boolean"], False)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["boolean"].name, "boolean")

        space.values["boolean"] = True
        self.assertEqual(space.get_value("boolean"), True)
        self.assertEqual(space.values, {"boolean": True})
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["boolean"].name, "boolean")

    def test_choice(self):
        space = ts.TunableSpace()
        choice = space.choice("choice", [1, 2, 3, 4], default=4)
        self.assertEqual(space.values["choice"], 4)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["choice"].name, "choice")

        space.values["choice"] = 2
        self.assertEqual(space.get_value("choice"), 2)
        self.assertEqual(space.values, {"choice": 2})
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["choice"].name, "choice")

    def test_int_range(self):
        space = ts.TunableSpace()
        int_range = space.int_range("int_range", start=1, stop=4, default=2)
        self.assertEqual(space.values["int_range"], 2)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["int_range"].name, "int_range")

        space.values["int_range"] = 3
        self.assertEqual(space.get_value("int_range"), 3)
        self.assertEqual(space.values, {"int_range": 3})
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["int_range"].name, "int_range")

    def test_float_range(self):
        space = ts.TunableSpace()
        float_range = space.float_range(
            "float_range", start=0.4, stop=4.4, default=2.0
        )
        self.assertEqual(space.values["float_range"], 2.0)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["float_range"].name, "float_range")

        space.values["float_range"] = 3.0
        self.assertEqual(space.get_value("float_range"), 3.0)
        self.assertEqual(space.values, {"float_range": 3.0})
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["float_range"].name, "float_range")

    def test_variables(self):
        space = ts.TunableSpace()
        choice = space.choice("choice", [1, 2, 3, 4], default=4)
        self.assertEqual(space.values["choice"], 4)
        self.assertEqual(len(space.variables), 1)
        self.assertEqual(space.variables["choice"].name, "choice")

        int_range = space.int_range("int_range", start=1, stop=4, default=2)
        self.assertEqual(space.values["int_range"], 2)
        self.assertEqual(len(space.variables), 2)
        self.assertEqual(space.variables["int_range"].name, "int_range")

    def test_not_populated_variable(self):
        space = ts.TunableSpace()
        choice = space.choice("choice", [1, 2, 3, 4], default=2)
        self.assertEqual(choice, 2)

    def test_populated_variable(self):
        space = ts.TunableSpace()
        space.values["choice"] = 2
        choice = space.choice("choice", [1, 2, 3, 4], default=4)
        self.assertEqual(choice, 2)

        space["choice"] = 3
        self.assertNotEqual(space.values["choice"], 2)
        self.assertEqual(space.values["choice"], 3)

    def test_state(self):
        space = ts.TunableSpace()
        choice = space.choice("choice", [1, 2, 3, 4], default=4)
        int_range = space.int_range("int_range", start=1, stop=4, default=2)

        new_space = space.from_state(space.get_state())
        self.assertEqual(new_space.get_value("choice"), 4)
        self.assertEqual(new_space.get_value("int_range"), 2)
        self.assertEqual(len(new_space.variables), 2)
        self.assertEqual(len(new_space.values), 2)

        self.assertEqual(new_space.variables["choice"].name, "choice")
        self.assertEqual(new_space.variables["choice"].default, 4)
        self.assertEqual(new_space.variables["choice"].values, [1, 2, 3, 4])

        self.assertEqual(new_space.variables["int_range"].name, "int_range")
        self.assertEqual(new_space.variables["int_range"].default, 2)
        self.assertEqual(new_space.variables["int_range"].start, 1)
        self.assertEqual(new_space.variables["int_range"].stop, 4)
        self.assertEqual(new_space.variables["int_range"].step, 1)
        self.assertEqual(new_space.variables["int_range"].endpoint, False)

    def test_exception(self):
        space = ts.TunableSpace()
        flag = True
        try:
            val = space.get_value("test")
            flag = False
        except:
            pass
        self.assertTrue(flag)


if __name__ == "__main__":
    unittest.main()
