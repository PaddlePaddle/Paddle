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

import contextlib
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import core, framework
from paddle.fluid.dygraph.static_runner import DescParser


@contextlib.contextmanager
def program_scope_guard():
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    with fluid.scope_guard(scope):
        with fluid.program_guard(prog, startup_prog):
            with fluid.unique_name.guard():
                yield


# NOTE: Because RunProgramOp has a special output of type std::vector<Scope *>, 
# the OpTest cannot be used in RunProgramOp. The variable type cannot be specified
# when creating output variables in OpTest, default type is LoDTensor
class RunProgramOpTest(unittest.TestCase):
    def check_output(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            # TODO: RunProgramOp is not recommended for use in static mode now
            self.check_output_dygraph(place)

    def check_output_dygraph(self, place):
        def create_var_base(is_input, name, np_value):
            if is_input:
                return core.VarBase(
                    value=np_value, name=name, place=place, zero_copy=True)
            return framework._varbase_creator(dtype=None, shape=None, name=name)

        # Step 1. run op
        with fluid.dygraph.guard(place):
            # build inputs
            inputs = {}
            inputs['X'] = []
            for (name, np_value) in self.inputs['X']:
                inputs['X'].append(create_var_base(True, name, np_value))
            inputs['Params'] = []
            for (name, np_value) in self.inputs['Params']:
                inputs['Params'].append(create_var_base(True, name, np_value))

            # build outputs
            outputs = {}
            outputs['Out'] = []
            for (name, np_value) in self.outputs['Out']:
                outputs['Out'].append(create_var_base(False, name, np_value))
            outputs['OutScope'] = framework._varbase_creator(
                type=core.VarDesc.VarType.STEP_SCOPES,
                name="program_out_scope",
                persistable=True)
            inner_scope = core.Scope()
            outputs['OutScope'].value().set_scope(inner_scope)

            framework._dygraph_tracer().trace_op(
                type=self.op_type,
                inputs=inputs,
                outputs=outputs,
                attrs=self.attrs)

        # Step 2. compare output
        expect_outs = self.outputs['Out']
        for i, out in enumerate(outputs['Out']):
            self.assertTrue(
                np.allclose(
                    out.numpy(), expect_outs[i][1], atol=1e-5))


class TestRunProgramOpWithFC(RunProgramOpTest):
    class MatrixGenerate:
        def __init__(self, mb, ic, oc, h, w, dtype):
            self.input = np.random.random((mb, ic, h, w)).astype(dtype)
            self.weights = np.random.random((ic * h * w, oc)).astype(dtype)
            self.bias = np.random.random((1, oc)).astype(dtype)

    def setUp(self):
        self.op_type = "run_program"
        self.dtype = np.float64

        self.matrix = self.MatrixGenerate(1, 1, 10, 28, 28, self.dtype)
        self.inputs = {
            'X': [('img', self.matrix.input)],
            'Params':
            [('fc_0.w_0', self.matrix.weights), ('fc_0.b_0', self.matrix.bias)]
        }

        self.program_desc, self.fwd_op_num = self.simple_fc_program_desc()

        self.attrs = {
            'global_block': self.program_desc.block(0),
            'start_op_index': 0,
            'end_op_index': self.fwd_op_num
        }

        self.outputs = {'Out': [('fc_0.tmp_2', self.fc_refer(self.matrix))], }

    def test_check_output(self):
        self.check_output()

    def simple_fc_program_desc(self):
        with program_scope_guard():
            main_program = fluid.default_main_program()
            img = fluid.data(
                name='img', shape=[None, 1, 28, 28], dtype='float32')
            pred = fluid.layers.fc(input=img, size=10, act='relu')
            fwd_op_num = main_program.global_block().desc.op_size()
            fluid.backward.gradients(targets=[pred], inputs=[img])
            return main_program.desc, fwd_op_num

    def fc_refer(self, matrix):
        in_n, in_c, in_h, in_w = matrix.input.shape
        w_i, w_o = matrix.weights.shape

        x_data = np.reshape(matrix.input, [in_n, in_c * in_h * in_w])
        w_data = np.reshape(matrix.weights, [w_i, w_o])
        b_data = np.reshape(matrix.bias, [1, w_o])

        result = np.dot(x_data, w_data) + b_data
        return np.maximum(result, 0)


if __name__ == "__main__":
    unittest.main()
