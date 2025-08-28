# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import _legacy_C_ops
from paddle.base import core
from paddle.base.dygraph.base import switch_to_static_graph
from paddle.base.framework import Variable


def _append_backward_desc(main_program, outs):
    # make sure all status of is_test are False in train mode.
    program = main_program.clone()
    targets = []
    for out in outs:
        if isinstance(out, Variable):
            targets.append(program.global_block().var(out.name))

    if targets:
        paddle.base.backward.gradients(targets=targets, inputs=[])

    return program


# def _set_grad_type(params, train_program):
#     # NOTE: if user set sparse gradient mode, the param's gradient
#     # will be SelectedRows, not DenseTensor. But tracer will just
#     # set param grad Tensor by forward Tensor(DenseTensor)
#     # If we don't change grad_var type here, RunProgramOp need
#     # transform SelectedRows to DenseTensor forcibly, it may not
#     # be user wanted result.
#     for param in params:
#         grad_name = param.name + core.grad_var_suffix()
#         grad_var = train_program.desc.block(0).find_var(
#             grad_name.encode())
#         # NOTE: cannot find var desc maybe no problem, such as in batch_norm
#         if grad_var is None:
#             continue
#         param._set_grad_type(grad_var.type())


def _create_out(var):
    assert isinstance(var, Variable)
    var_desc = var.desc
    out = core.eager.Tensor(
        var_desc.dtype(),
        var_desc.shape(),
        var_desc.name(),
        var_desc.type(),
        False,
    )
    out.stop_gradient = False
    return out


@switch_to_static_graph
def _add_build_strategy_for(input_program, start_op_index, end_op_index):
    compiled_program = paddle.static.CompiledProgram(
        core.Graph(input_program.desc, start_op_index, end_op_index),
        build_strategy=paddle.static.BuildStrategy(),
    )
    compiled_program._compile(
        core.Scope(), paddle.framework._current_expected_place()
    )
    ir_graph = paddle.base.framework.IrGraph(compiled_program._graph)
    built_program = ir_graph.to_program()
    return built_program


class TestRunProgram(unittest.TestCase):
    def test_eager(self):
        paddle.set_device('cpu')
        paddle.enable_static()
        # step 1: construct program
        x = paddle.static.data(shape=[2, 4], name='x')
        x.stop_gradient = False
        y = paddle.static.data(shape=[4, 2], name='y')
        y.stop_gradient = False
        out = paddle.matmul(x, y)

        main_program = paddle.static.default_main_program()
        program = _append_backward_desc(main_program, [out])
        forward_program = _add_build_strategy_for(
            program, 0, main_program.desc.block(0).op_size()
        )
        backward_program = _add_build_strategy_for(
            program,
            main_program.desc.block(0).op_size() + 1,
            program.desc.block(0).op_size(),
        )

        paddle.disable_static('cpu')
        # step 2: call run_program in eager mode
        x_t = paddle.ones([2, 4])
        x_t.name = "x"
        x_t.stop_gradient = False
        y_t = paddle.ones([4, 2])
        y_t.name = "y"
        y_t.stop_gradient = False

        out_t = _create_out(out)

        scope = core.Scope()
        attrs = [
            'global_block',
            program.desc.block(0),
            'start_op_index',
            0,
            'end_op_index',
            main_program.desc.block(0).op_size(),
            'is_test',
            False,
            'program_id',
            paddle.utils._hash_with_id(program),
            'param_grad_names',
            [],
            'out_grad_names',
            [out.name + '@GRAD'],
            'x_grad_names',
            [x_t.name + '@GRAD', y_t.name + '@GRAD'],
            'x_names',
            [x_t.name, y_t.name],
        ]

        use_interpretorcore = True
        attrs.extend(('use_interpretorcore', use_interpretorcore))
        if use_interpretorcore:
            attrs.extend(
                (
                    'forward_global_block',
                    forward_program.desc.block(0),
                    'backward_global_block',
                    backward_program.desc.block(0),
                )
            )

        _legacy_C_ops.run_program(
            [x_t, y_t], None, [out_t], [scope], None, *attrs
        )

        loss = paddle.mean(out_t)
        loss.backward()

        np.testing.assert_array_equal(np.ones([2, 2]) * 4, out_t.numpy())
        np.testing.assert_array_equal(np.ones([2, 4]) * 0.5, x_t.grad.numpy())
        np.testing.assert_array_equal(np.ones([4, 2]) * 0.5, y_t.grad.numpy())


if __name__ == '__main__':
    unittest.main()
