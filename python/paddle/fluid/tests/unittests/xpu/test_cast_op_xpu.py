#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import sys

sys.path.append("..")
import unittest
import op_test
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

typeid_dict = {
    'int32': int(core.VarDesc.VarType.INT32),
    'int64': int(core.VarDesc.VarType.INT64),
    'float32': int(core.VarDesc.VarType.FP32),
    'float16': int(core.VarDesc.VarType.FP16),
    'bool': int(core.VarDesc.VarType.BOOL),
}


def create_test_class(in_typename, out_typename):
    class Cls(op_test.OpTest):
        def setUp(self):
            ipt = np.random.random(size=[10, 10])
            self.inputs = {'X': ipt.astype(in_typename)}
            self.outputs = {'Out': ipt.astype(in_typename).astype(out_typename)}
            self.attrs = {
                'in_dtype': typeid_dict[in_typename],
                'out_dtype': typeid_dict[out_typename],
            }
            self.op_type = 'cast'
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

    cls_name = "cast_{0}_{1}".format(in_typename, out_typename)
    Cls.__name__ = cls_name
    globals()[cls_name] = Cls


for in_type in {'float16', 'float32', 'int32', 'int64', 'bool'}:
    for out_type in {'float16', 'float32', 'int32', 'int64'}:
        create_test_class(in_type, out_type)


class TestCastOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.XPUPlace(0))
            self.assertRaises(TypeError, fluid.layers.cast, x1, 'int32')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
