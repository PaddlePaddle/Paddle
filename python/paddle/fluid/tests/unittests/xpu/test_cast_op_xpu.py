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

<<<<<<< HEAD
=======
from __future__ import print_function
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
import sys

sys.path.append("..")
import unittest
<<<<<<< HEAD

import numpy as np
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
=======
import op_test
import numpy as np
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from op_test_xpu import XPUOpTest

from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

typeid_dict = {
    'int32': int(core.VarDesc.VarType.INT32),
    'int64': int(core.VarDesc.VarType.INT64),
    'float32': int(core.VarDesc.VarType.FP32),
    'float16': int(core.VarDesc.VarType.FP16),
    'bool': int(core.VarDesc.VarType.BOOL),
    'uint8': int(core.VarDesc.VarType.UINT8),
<<<<<<< HEAD
    'float64': int(core.VarDesc.VarType.FP64),
=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
}


class XPUTestCastOp(XPUOpTestWrapper):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self):
        self.op_name = 'cast'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestCastOp
        classes = []
<<<<<<< HEAD
        for out_type in {
            'float16',
            'float32',
            'int32',
            'int64',
            'uint8',
            'bool',
            'float64',
        }:
=======
        for out_type in {'float16', 'float32', 'int32', 'int64', 'uint8'}:
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            class_name = 'XPUTestCastOp_outtype_' + out_type
            attr_dict = {'out_typename': out_type}
            classes.append([class_name, attr_dict])
        return base_class, classes

    class TestCastOp(XPUOpTest):
<<<<<<< HEAD
        def setUp(self):
            ipt = np.random.random(size=[10, 10])
            in_typename = self.in_type_str
            out_typename = (
                'float32'
                if not hasattr(self, 'out_typename')
                else self.out_typename
            )
=======

        def setUp(self):
            ipt = np.random.random(size=[10, 10])
            in_typename = self.in_type_str
            out_typename = 'float32' if not hasattr(
                self, 'out_typename') else self.out_typename
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.inputs = {'X': ipt.astype(in_typename)}
            self.outputs = {'Out': ipt.astype(in_typename).astype(out_typename)}
            self.attrs = {
                'in_dtype': typeid_dict[in_typename],
                'out_dtype': typeid_dict[out_typename],
            }
            self.op_type = 'cast'
            self.__class__.no_need_check_grad = True

        def test_check_output(self):
            self.check_output()


support_types = get_xpu_op_support_types('cast')
for stype in support_types:
    create_test_class(globals(), XPUTestCastOp, stype)


class TestCastOpError(unittest.TestCase):
<<<<<<< HEAD
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.XPUPlace(0)
            )
=======

    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of cast_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.XPUPlace(0))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self.assertRaises(TypeError, fluid.layers.cast, x1, 'int32')


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
