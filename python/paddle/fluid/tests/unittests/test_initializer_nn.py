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
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.nn.initializer as initializer
from paddle.fluid.core import VarDesc

DELTA = 0.00001


def check_cast_op(op):
    return op.type == 'cast' and \
           op.attr('in_dtype') == VarDesc.VarType.FP32 and \
           op.attr('out_dtype') == VarDesc.VarType.FP16


class TestConstantInitializer(unittest.TestCase):
    def static_test_constant_initializer_common(self,
                                                init_inst,
                                                dtype="float32",
                                                value_target=0.0):
        paddle.enable_static()
        program = framework.Program()
        block = program.global_block()
        for _ in range(2):
            block.create_parameter(
                dtype=dtype,
                shape=[5, 10],
                lod_level=0,
                name="param",
                initializer=init_inst)
        num_ops = 2 if dtype == "float16" else 1
        self.assertEqual(len(block.ops), num_ops)
        init_op = block.ops[0]
        self.assertEqual(init_op.type, 'fill_constant')
        self.assertAlmostEqual(init_op.attr('value'), value_target, delta=DELTA)
        paddle.disable_static()
        return block

    def test_constant_initializer_default_value_static(self, dtype="float32"):
        """Test the constant initializer with default value in static graph
        """
        block = self.static_test_constant_initializer_common(
            init_inst=initializer.Constant(), dtype=dtype, value_target=0.0)
        return block

    def test_constant_initializer_default_value_dygraph(self, dtype="float32"):
        """Test constant initializer with supplied value in dygraph
        """
        with fluid.dygraph.guard():
            linear = nn.Linear(2, 4, weight_attr=nn.initializer.Constant())
            mat_target = np.ones((2, 4), dtype=dtype) * 0.0
            mat_linear = linear.weight.numpy()
            mismatch = np.sum(
                (mat_target - mat_linear) * (mat_target - mat_linear))
            self.assertAlmostEqual(mismatch, 0.0, delta=DELTA)

    def test_constant_initializer_static(self, dtype="float32"):
        """Test constant initializer with supplied value in static graph
        """
        block = self.static_test_constant_initializer_common(
            init_inst=initializer.Constant(2.3), dtype=dtype, value_target=2.3)
        return block

    def test_constant_initializer_dygraph(self, dtype="float32"):
        """Test constant initializer with supplied value in dygraph
        """
        with fluid.dygraph.guard():
            linear = nn.Linear(
                2, 4, weight_attr=nn.initializer.Constant(value=2.0))
            mat_target = np.ones((2, 4), dtype=dtype) * 2.0
            mat_linear = linear.weight.numpy()
            mismatch = np.sum(
                (mat_target - mat_linear) * (mat_target - mat_linear))
            self.assertAlmostEqual(mismatch, 0.0, delta=DELTA)

    def test_constant_initializer_fp16(self):
        """Test constant initializer with float16
        """
        block = self.test_constant_initializer_default_value_static("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        block = self.test_constant_initializer_static("float16")
        self.assertTrue(check_cast_op(block.ops[1]))
        self.test_constant_initializer_default_value_dygraph("float16")
        self.test_constant_initializer_dygraph("float16")


if __name__ == '__main__':
    unittest.main()
