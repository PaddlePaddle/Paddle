#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle.fluid import core




from op import Operator

paddle.enable_static()


class XPUTestFillOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'fill'
        self.use_dynamic_create_class = False

    class TestFillOp1(XPUOpTest):
        def setUp(self):
            self.op_type = "fill"
            val = np.random.random(size=[100, 200])
            self.inputs = {}
            self.attrs = {
                'value': val.flatten().tolist(),
                'shape': [100, 200],
                'dtype': int(core.VarDesc.VarType.FP64),
                'force_cpu': False,
            }
            self.outputs = {'Out': val.astype('float64')}

        def test_check_output(self):
            self.check_output_with_place(paddle.XPUPlace(0))

    class TestFillOp2(XPUOpTest):
        def setUp(self):
            self.op_type = "fill"
            val = np.random.random(size=[100, 200])
            self.inputs = {}
            self.attrs = {
                'value': val.flatten().tolist(),
                'shape': [100, 200],
                'dtype': int(core.VarDesc.VarType.FP64),
                'force_cpu': True,
            }
            self.outputs = {'Out': val.astype('float64')}

        def test_check_output(self):
            self.check_output()

    class TestFillOp3(unittest.TestCase):
        def check_with_place(self, place, f_cpu):
            scope = core.Scope()
            # create Out Variable
            out = scope.var('Out').get_tensor()

            # create and run fill_op operator
            val = np.random.random(size=[300, 200])
            fill_op = Operator(
                "fill",
                value=val.flatten(),
                shape=[300, 200],
                dtype=int(core.VarDesc.VarType.FP32),
                force_cpu=f_cpu,
                Out='Out',
            )
            fill_op.run(scope, place)

            # get result from Out
            result_array = np.array(out)
            full_array = np.array(val, 'float32')

            np.testing.assert_array_equal(result_array, full_array)

        def test_fill_op(self):
            places = [core.CPUPlace()]
            if core.is_compiled_with_xpu():
                places.append(core.XPUPlace(0))

            for place in places:
                self.check_with_place(place, True)
                self.check_with_place(place, False)


support_types = get_xpu_op_support_types('fill')
for stype in support_types:
    create_test_class(globals(), XPUTestFillOp, stype)

if __name__ == '__main__':
    unittest.main()
