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

from __future__ import print_function

import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid


def check_if_mkldnn_primitives_exist_in_bwd(test_case, op_type, x, out,
                                            out_grad, x_grad):
    def __assert_close(tensor, np_array, msg, atol=1e-4):
        test_case.assertTrue(
            np.allclose(
                np.array(tensor), np_array, atol=atol), msg)

    place = core.CPUPlace()

    var_dict = {'x': x, 'out': out, 'out@GRAD': out_grad, 'x@GRAD': x_grad}
    var_names = list(var_dict.keys())
    ground_truth = {name: var_dict[name] for name in var_names}

    program = fluid.Program()
    with fluid.program_guard(program):
        block = program.global_block()
        for name in ground_truth:
            block.create_var(
                name=name, dtype=np.float32, shape=ground_truth[name].shape)

        op = block.append_op(
            type=op_type,
            inputs={'X': block.var('x'), },
            outputs={'Out': block.var('out')},
            attrs={'use_mkldnn': True})

        # Generate backward op_desc
        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(op.desc,
                                                                  set(), [])
        grad_op_desc = grad_op_desc_list[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc)
        for var_name in grad_op_desc.output_arg_names():
            block.desc.var(var_name.encode('ascii'))
        grad_op_desc.infer_var_type(block.desc)
        grad_op_desc.infer_shape(block.desc)
        for arg in grad_op_desc.output_arg_names():
            grad_var = block.desc.find_var(arg.encode('ascii'))
            grad_var.set_dtype(core.VarDesc.VarType.FP32)

        exe = fluid.Executor(place)

        # Do at least 2 iterations
        for i in range(2):
            out = exe.run(
                program,
                feed={name: var_dict[name]
                      for name in ['x', 'out@GRAD']},
                fetch_list=['x@GRAD', 'out'])

        __assert_close(x_grad, out[0], 'x@GRAD')
