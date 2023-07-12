#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

from cinn.common import Bool, Float, Int, UInt, make_const


class TestType(unittest.TestCase):
    def test_type_constructs(self):
        self.assertEqual(str(Float(32)), "float32")
        self.assertEqual(str(Int(32)), "int32")
        self.assertEqual(str(Int(64)), "int64")
        self.assertEqual(str(UInt(64)), "uint64")
        self.assertEqual(str(UInt(32)), "uint32")
        self.assertEqual(str(Bool()), "bool")

    def test_make_const(self):
        self.assertEqual(str(make_const(Float(32), 1.23)), "1.23000002f")
        self.assertEqual(str(make_const(Int(32), 1.23)), "1")
        # self.assertEqual(str(make_const(UInt(32), 1.23)), "1")


if __name__ == "__main__":
    unittest.main()
