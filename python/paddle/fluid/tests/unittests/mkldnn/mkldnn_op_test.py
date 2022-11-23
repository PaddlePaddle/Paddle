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

import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid


def __assert_close(test_case, tensor, np_array, msg, atol=1e-4):
    test_case.assertTrue(np.allclose(np.array(tensor), np_array, atol=atol),
                         msg)


def check_if_mkldnn_primitives_exist_in_bwd(test_case, op_type, x, out,
                                            out_grad, x_grad):
    place = core.CPUPlace()

    var_dict = {'x': x, 'out': out, 'out@GRAD': out_grad, 'x@GRAD': x_grad}
    var_names = list(var_dict.keys())
    ground_truth = {name: var_dict[name] for name in var_names}

    program = fluid.Program()
    with fluid.program_guard(program):
        block = program.global_block()
        for name in ground_truth:
            block.create_var(name=name,
                             dtype=np.float32,
                             shape=ground_truth[name].shape)

        op = block.append_op(type=op_type,
                             inputs={
                                 'X': block.var('x'),
                             },
                             outputs={'Out': block.var('out')},
                             attrs={'use_mkldnn': True})

        # Generate backward op_desc
        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
            op.desc, set(), [])
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

        __assert_close(test_case, x_grad, out[0], 'x@GRAD')


def check_if_mkldnn_batchnorm_primitives_exist_in_bwd(test_case, var_dict,
                                                      place, shape,
                                                      data_layout):

    var_names = [
        'x', 'scale', 'bias', 'mean', 'variance', 'y', 'saved_mean',
        'saved_variance'
    ]
    ground_truth = {name: var_dict[name] for name in var_names}
    program = fluid.Program()
    with fluid.program_guard(program):
        block = program.global_block()
        for name in ground_truth:
            block.create_var(name=name,
                             dtype='float32',
                             shape=ground_truth[name].shape)
        bn_op = block.append_op(
            type="batch_norm",
            inputs={
                "X": block.var('x'),
                "Scale": block.var('scale'),
                "Bias": block.var('bias'),
                "Mean": block.var('mean'),
                "Variance": block.var('variance')
            },
            outputs={
                "Y": block.var('y'),
                "MeanOut": block.var('mean'),  # share memory
                "VarianceOut": block.var('variance'),  # share memory
                "SavedMean": block.var('saved_mean'),
                "SavedVariance": block.var('saved_variance')
            },
            attrs={
                "momentum": test_case.momentum,
                "epsilon": test_case.epsilon,
                "is_test": False,
                "data_layout": data_layout,
                "use_mkldnn": test_case.use_mkldnn,
                "fuse_with_relu": test_case.fuse_with_relu,
                "use_global_stats": test_case.use_global_stats
            })
        block.create_var(name='y@GRAD',
                         dtype='float32',
                         shape=var_dict['y'].shape)

        # generate backward op_desc
        grad_op_desc_list, op_grad_to_var = core.get_grad_op_desc(
            bn_op.desc, test_case.no_grad_set, [])
        grad_op_desc = grad_op_desc_list[0]
        new_op_desc = block.desc.append_op()
        new_op_desc.copy_from(grad_op_desc)
        for var_name in grad_op_desc.output_arg_names():
            block.desc.var(var_name.encode("ascii"))
        grad_op_desc.infer_var_type(block.desc)
        grad_op_desc.infer_shape(block.desc)
        for arg in grad_op_desc.output_arg_names():
            grad_var = block.desc.find_var(arg.encode("ascii"))
            grad_var.set_dtype(core.VarDesc.VarType.FP32)
        program._sync_with_cpp()

        exe = fluid.Executor(place)
        # Do at least 2 iterations
        for i in range(2):
            out = exe.run(
                program,
                feed={
                    name: var_dict[name]
                    for name in
                    ['x', 'scale', 'bias', 'mean', 'variance', 'y@GRAD']
                },
                fetch_list=test_case.fetch_list)
            for id, name in enumerate(test_case.fetch_list):
                __assert_close(test_case, var_dict[name], out[id], name)

        print("MKLDNN op test forward passed: ", str(place), data_layout)


def format_reorder(out, size):
    in_n = size[0]
    out_h = size[2]
    out_w = size[3]
    out_c = size[1]
    out_tmp = np.zeros((in_n, out_h, out_w, out_c))
    for n in range(in_n):
        for i in range(out_h):
            for j in range(out_w):
                for m in range(out_c):
                    out_tmp[n, i, j, m] = out[n, m, i, j]
    return out_tmp.reshape(in_n, out_c, out_h, out_w)
