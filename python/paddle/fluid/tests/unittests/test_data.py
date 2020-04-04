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

from __future__ import print_function

import unittest

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import Program, program_guard


class TestApiDataError(unittest.TestCase):
    def test_fluid_data(self):
        with program_guard(Program(), Program()):

            # 1. The type of 'name' in fluid.data must be str.
            def test_name_type():
                fluid.data(name=1, shape=[2, 25], dtype="bool")

            self.assertRaises(TypeError, test_name_type)

            # 2. The type of 'shape' in fluid.data must be list or tuple.
            def test_shape_type():
                fluid.data(name='data1', shape=2, dtype="bool")

            self.assertRaises(TypeError, test_shape_type)

    def test_layers_data(self):
        with program_guard(Program(), Program()):

            # 1. The type of 'name' in layers.data must be str.
            def test_name_type():
                layers.data(name=1, shape=[2, 25], dtype="bool")

            self.assertRaises(TypeError, test_name_type)

            # 2. The type of 'shape' in layers.data must be list or tuple.
            def test_shape_type():
                layers.data(name='data1', shape=2, dtype="bool")

            self.assertRaises(TypeError, test_shape_type)


if __name__ == "__main__":
    unittest.main()
