# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.test_sum_op import TestSumOp
import numpy as np
import paddle.fluid.op as fluid_op


class TestSumMKLDNN(TestSumOp):

    def setUp(self):
        self.op_type = "sum"
        self.init_data_type()
        self.use_mkldnn = True
        x0 = np.random.random((25, 8)).astype(self.dtype)
        x1 = np.random.random((25, 8)).astype(self.dtype)
        x2 = np.random.random((25, 8)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_data_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_grad(['x0'], 'Out', check_dygraph=False)


class TestMKLDNNSumInplaceOp(unittest.TestCase):

    def setUp(self):
        self.op_type = "sum"
        self.init_data_type()
        self.use_mkldnn = True
        self.x0 = np.random.random((25, 8)).astype(self.dtype)
        self.x1 = np.random.random((25, 8)).astype(self.dtype)

    def init_data_type(self):
        self.dtype = np.float32

    def test_check_output(self):
        place = core.CPUPlace()
        scope = core.Scope()
        out_var_name = "x0"
        inputs = {"X": [("x0", self.x0), ("x1", self.x1)]}

        for input_key in inputs:
            for per_input in inputs[input_key]:
                var_name, var_value = per_input[0], per_input[1]
                var = scope.var(var_name)
                tensor = var.get_tensor()
                tensor.set(var_value, place)

        sum_op = fluid_op.Operator("sum",
                                   X=["x0", "x1"],
                                   Out=out_var_name,
                                   use_mkldnn=True)
        expected_out = np.array(self.x0 + self.x1)
        sum_op.run(scope, place)
        out = scope.find_var("x0").get_tensor()
        out_array = np.array(out)
        np.testing.assert_allclose(
            expected_out,
            out_array,
            rtol=1e-05,
            atol=1e-05,
            err_msg='Inplace sum_mkldnn_op output has diff with expected output'
        )

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    from paddle import enable_static
    enable_static()
    unittest.main()
