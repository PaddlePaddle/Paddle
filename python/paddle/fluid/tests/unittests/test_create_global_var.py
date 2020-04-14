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

import os, shutil
import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestCreateGlobalVarError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_shape():
                fluid.layers.create_global_var(1, 2.0, np.float32)

            self.assertRaises(TypeError, test_shape)

            def test_shape_item():
                fluid.layers.create_global_var([1.0, 2.0, 3.0], 2.0, 'float32')

            self.assertRaises(TypeError, test_shape_item)

            # Since create_global_var support all dtype in convert_dtype(). 
            # Hence, assertRaises ValueError not TypeError. 
            def test_dtype():
                fluid.layers.create_global_var([1, 2, 3], 2.0, np.complex128)

            self.assertRaises(ValueError, test_dtype)


if __name__ == '__main__':
    unittest.main()
