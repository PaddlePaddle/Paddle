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

from __future__ import print_function

import unittest
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.tests.unittests.op_test import OpTest

fluid.core.globals()['FLAGS_tensor_dump_ops'] = 'fc_mkldnn_fwd'
fluid.core.globals()['FLAGS_dump_limit'] = 64


def fully_connected_naive(input, weights, bias_data):
    result = np.dot(input, weights) + bias_data
    return result


class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w):
        self.input = np.random.random((mb, ic * h * w)).astype("float32")
        self.weights = np.random.random((ic * h * w, oc)).astype("float32")


class TestFCMKLDNNOp(OpTest):
    def create_data(self):
        self.matrix = MatrixGenerate(1, 10, 15, 3, 3)
        self.bias = np.random.random(15).astype("float32")

    def setUp(self):
        self.op_type = "fc"
        self._cpu_only = True
        self.use_mkldnn = True
        self.create_data()
        self.inputs = {
            'Input': self.matrix.input,
            'W': self.matrix.weights,
            'Bias': self.bias
        }

        self.attrs = {'use_mkldnn': self.use_mkldnn}

        self.outputs = {
            'Out': fully_connected_naive(self.matrix.input, self.matrix.weights,
                                         self.bias)
        }

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass


class TestFCMKLDNNOp1(TestFCMKLDNNOp):
    def create_data(self):
        self.matrix = MatrixGenerate(2, 15, 48, 2, 2)
        self.bias = np.random.random(48).astype("float32")


class TestTensorDump():
    def TestCoverage(self):
        # fluid.set_flags({'FLAGS_tensor_dump_ops': "fc_mkldnn_fwd=NHWC,transpose_mkldnn_fwd=NHWC"})
        # fluid.set_flags({'FLAGS_dump_limit': 64})
        # print("SUCCESS! set flags successfully!")
        # os.environ['TENSOR_DUMP_OPERATORS'] = "fc_mkldnn_fwd"
        # os.environ['TENSOR_DUMP_FOLDER'] = "out"
        # os.environ['DUMP_LIMIT'] = "64"
        self.op_type = "fc"
        self._cpu_only = True
        self.use_mkldnn = True
        self.x = np.random.random((2, 15 * 2 * 2)).astype("float32")
        self.w = np.random.random((48 * 2 * 2, 15)).astype("float32")
        self.bias = np.random.random(48).astype("float32")
        self.inputs = {'Input': self.x, 'W': self.w, 'Bias': self.bias}

        self.attrs = {'use_mkldnn': self.use_mkldnn}

        self.outputs = fully_connected_naive(self.x, self.w, self.bias)

        place = core.CPUPlace()
        var_dict = {
            'x': self.x,
            'w': self.w,
            'bias': self.bias,
            'out': self.outputs
        }

        program = fluid.Program()
        with fluid.program_guard(program):
            block = program.global_block()
            for name in var_dict:
                block.create_var(
                    name=name, dtype=np.float32, shape=var_dict[name].shape)

            op = block.append_op(
                type=self.op_type,
                inputs={
                    'Input': block.var('x'),
                    'W': block.var('w'),
                    'Bias': block.var('bias')
                },
                outputs={'Out': block.var('out')},
                attrs={'use_mkldnn': True})

        exe = fluid.Executor(place)
        # Do at least 2 iterations
        for i in range(2):
            out = exe.run(
                program,
                # feed={'Input','Bias'
                # },
                # fetch_list=test_case.fetch_list
            )
            # for id, name in enumerate(test_case.fetch_list):
            #     __assert_close(test_case, var_dict[name], out[id], name)

        # def test_global_dump(self):
        #     os.environ['TENSOR_DUMP_OPERATORS'] = 'fc_mkldnn_fwd=NHWC,transpose_mkldnn_fwd=NHWC'
        #     os.environ['TENSOR_DUMP_FOLDER'] = 'out'
        #     os.environ['DUMP_LIMIT'] = '64'

        #     self.input = np.random.random((1, 10 * 3 * 3)).astype("float32")
        #     self.weights = np.random.random((10 * 3 * 3, 15)).astype("float32")
        #     self.bias = np.random.random(15).astype("float32")
        #     self.op_type = "fc"
        #     self._cpu_only = True
        #     self.use_mkldnn = True
        #     self.inputs = {
        #         'Input': self.input,
        #         'W': self.weights,
        #         'Bias': self.bias
        #     }
        #     self.attrs = {'use_mkldnn': self.use_mkldnn}

        #     self.outputs = {
        #         'Out': fully_connected_naive(self.input, self.weights,
        #                                      self.bias)
        #     }
        #     self.check_output(check_dygraph=False)
        #     print("Tested")

        # x = np.random.random(1, 3, 2, 2).astype(np.float32)
        # weights = np.random.random(1, 15, 2, 2).astype(np.float32)
        # bias = np.random.random(1, 15).astype(np.float32)
        # target = x * weights + bias
        # weights = np.random.random()
        # ground_truth = {}
        # program = fluid.Program()
        # with fluid.program_guard(program):
        #     block = program.global_block()
        #     for name in ground_truth:

        #     op = block.append_op(
        #         type = "fc",
        #         inputs = {'X': block.var('x'), 'Filter'}

        #     )


if __name__ == "__main__":
    unittest.main()
