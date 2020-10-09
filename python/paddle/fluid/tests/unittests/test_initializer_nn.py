#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.nn.initializer as initializer
from paddle.fluid.core import VarDesc

paddle.enable_static()

DELTA = 0.00001


def check_cast_op(op):
    return op.type == 'cast' and \
           op.attr('in_dtype') == VarDesc.VarType.FP32 and \
           op.attr('out_dtype') == VarDesc.VarType.FP16


class TestConstantInitializer(unittest.TestCase):
    def test_constant_initializer_default_value(self, dtype="float32"):
        """Test the constant initializer with default value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.Constant())
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 0.0, delta=DELTA)
        return block

    def test_constant_initializer(self, dtype="float32"):
        """Test constant initializer with supplied value
        """
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=initializer.Constant(2.3))
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), 2.3, delta=DELTA)
        return block

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16
        """
        block = self.test_constant_initializer_default_value("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_constant_initializer("float16")
        self.assertTrue(check_cast_op(block.ops[1]))


if __name__ == '__main__':
    unittest.main()
