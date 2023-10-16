# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


# run this: FLAGS_enable_new_ir_in_executor=True python test_program_prim.py or
#           FLAGS_enable_pir_api=True FLAGS_enable_new_ir_in_executor=True python test_program_prim.py
import unittest

import numpy as np

import paddle
from paddle import pir
from paddle.base import core
from paddle.decomposition.decomp import (
    decompose_bwd_op,
    decompose_fwd_op,
    get_graph_outputs_infos,
    related_graph_outputs,
    replace_graph_outputs,
)

paddle.enable_static()


def check_param_mappings(param_mappings):
    for VarDesc, Values in param_mappings.items():
        if len(Values) < 0 or len(Values) > 1:
            raise ValueError("currently only support one-to-one param_mappings")


def get_new_ir_grad_var_to_var_map(param_mappings, old_ir_grad_var_to_var_map):
    new_ir_grad_var_to_var_map = {}
    for grad_var, var in old_ir_grad_var_to_var_map.items():
        new_grad_var = param_mappings[grad_var][0].get_opresult()
        new_var = param_mappings[var][0].get_opresult()
        new_ir_grad_var_to_var_map[new_grad_var] = new_var
    return new_ir_grad_var_to_var_map


def get_fwd_op(bwd_op, grad_var_to_var_map):
    bwd_op_input_names = bwd_op.get_input_names()
    for idx, input_name in enumerate(bwd_op_input_names):
        if input_name == "out_grad":
            out_grad = bwd_op.operand(idx).source()
            out = grad_var_to_var_map[out_grad]
            fwd_op = out.get_defining_op()
            return fwd_op

    return None


def get_gelu_pir_program_and_param_map():
    shape = [2, 3]
    mp = paddle.static.Program()
    with paddle.static.program_guard(mp):
        # construct forward graph
        x = paddle.static.data('x', shape, dtype='float32')
        x.stop_gradient = False
        y = paddle.static.data('y', shape, dtype='float32')
        y.stop_gradient = False
        z = paddle.static.data('z', shape, dtype='float32')
        z.stop_gradient = False
        tmp1 = paddle.add(x, y)
        tmp2 = paddle.multiply(tmp1, z)
        out = paddle.nn.functional.gelu(tmp2, approximate=False)
        # construct backward graph
        gradients = paddle.static.gradients(out, [x, y, z])
        # get the old_ir_grad_var_to_var map
        old_ir_grad_var_to_var_map = {
            'gelu_0.tmp_0@GRAD': 'gelu_0.tmp_0',
            'elementwise_mul_0@GRAD': 'elementwise_mul_0',
            'elementwise_add_0@GRAD': 'elementwise_add_0',
            'z@GRAD': 'z',
            'x@GRAD': 'x',
            'y@GRAD': 'y',
        }

    newir_program, param_mappings = pir.translate_to_new_ir_with_param_map(
        mp.desc
    )
    check_param_mappings(param_mappings)
    new_ir_grad_var_to_var_map = get_new_ir_grad_var_to_var_map(
        param_mappings, old_ir_grad_var_to_var_map
    )

    return newir_program, new_ir_grad_var_to_var_map


class TestDecomposeOp1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [2, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [2, 3]
        self.z = np.random.random(self.shape_z).astype("float32")
        print("x: ", self.x)
        print("y: ", self.y)
        print("z: ", self.z)

    def gelu_net(self, flag=None):
        (
            newir_program,
            grad_var_to_var_map,
        ) = get_gelu_pir_program_and_param_map()

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            core._set_prim_forward_enabled(True)
            core._set_prim_backward_enabled(True)

            newir_ops = newir_program.global_block().ops
            decompose_bwd_ops_names = [
                "pd_op.gelu_grad",
                "pd_op.add_grad",
                "pd_op.multiply_grad",
            ]

            global_outputs = [newir_ops[5].result(0)]
            global_grads = [
                newir_ops[-1].result(0),
                newir_ops[-1].result(1),
                newir_ops[-2].result(1),
            ]

            # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
            (
                related_fwd_ops,
                related_fwd_ops_output_indexes,
            ) = get_graph_outputs_infos(
                newir_program.global_block(), global_outputs
            )  # without update during execution
            (
                related_bwd_ops,
                related_bwd_ops_output_indexes,
            ) = get_graph_outputs_infos(
                newir_program.global_block(), global_grads
            )

            for bwd_op in newir_ops:
                if (
                    flag == "decompose"
                    and bwd_op.name() in decompose_bwd_ops_names
                ):
                    fwd_op = get_fwd_op(bwd_op, grad_var_to_var_map)
                    fwd_inputs = [x.source() for x in fwd_op.operands()]
                    fwd_outputs = fwd_op.results()

                    # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                    if not core.has_custom_vjp(fwd_op):
                        related_bwd_op_index = related_graph_outputs(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            newir_program.global_block(),
                            bwd_op,
                            grad_var_to_var_map,
                            fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_graph_outputs(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

                        related_fwd_op_index = related_graph_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            newir_program.global_block(),
                            fwd_op,
                            grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_graph_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                    # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                    else:
                        related_fwd_op_index = related_graph_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            newir_program.global_block(),
                            fwd_op,
                            grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_graph_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                        related_bwd_op_index = related_graph_outputs(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            newir_program.global_block(),
                            bwd_op,
                            grad_var_to_var_map,
                            new_fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_graph_outputs(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

            # execution
            print("final graph: ", newir_program)
            exe = paddle.static.Executor()
            outs = exe.run(
                newir_program,
                feed={'x': self.x, 'y': self.y, 'z': self.z},
                fetch_list=[
                    global_outputs[0],
                    global_grads[0],
                    global_grads[1],
                    global_grads[2],
                ],
            )
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

        return outs

    def test_decompose_gelu_op(self):
        res = self.gelu_net("decompose")


def get_layer_norm_pir_program_and_param_map():
    shape = [2, 3]
    mp = paddle.static.Program()
    with paddle.static.program_guard(mp):
        # construct graph
        x = paddle.static.data('x', shape, dtype='float32')
        x.stop_gradient = False
        y = paddle.static.data('y', shape, dtype='float32')
        y.stop_gradient = False
        z = paddle.static.data('z', shape, dtype='float32')
        z.stop_gradient = False
        tmp1 = paddle.add(x, y)
        tmp2 = paddle.multiply(tmp1, z)
        scale = paddle.tensor.fill_constant(
            shape=tmp2.shape[1:],
            dtype=tmp2.dtype,
            value=1.0,
        )
        scale.stop_gradient = True
        bias = paddle.tensor.fill_constant(
            shape=tmp2.shape[1:],
            dtype=tmp2.dtype,
            value=2.0,
        )
        bias.stop_gradient = False
        out = paddle.nn.functional.layer_norm(
            tmp2, tmp2.shape[1:], scale, bias, 1e-5
        )
        # construct backward graph
        gradients = paddle.static.gradients(out, [x, y, z])

        # get the old_ir_grad_var_to_var map
        old_ir_grad_var_to_var_map = {
            'layer_norm_0.tmp_2@GRAD': 'layer_norm_0.tmp_2',
            "fill_constant_3.tmp_0@GRAD": "fill_constant_3.tmp_0",
            'elementwise_mul_1@GRAD': 'elementwise_mul_1',
            'elementwise_add_1@GRAD': 'elementwise_add_1',
            'z@GRAD': 'z',
            'x@GRAD': 'x',
            'y@GRAD': 'y',
        }

    newir_program, param_mappings = pir.translate_to_new_ir_with_param_map(
        mp.desc
    )
    check_param_mappings(param_mappings)
    new_ir_grad_var_to_var_map = get_new_ir_grad_var_to_var_map(
        param_mappings, old_ir_grad_var_to_var_map
    )

    return newir_program, new_ir_grad_var_to_var_map


class TestDecomposeOp2(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [2, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [2, 3]
        self.z = np.random.random(self.shape_z).astype("float32")
        print("x: ", self.x)
        print("y: ", self.y)
        print("z: ", self.z)

    def layer_norm_net(self, flag=None):
        (
            newir_program,
            grad_var_to_var_map,
        ) = get_layer_norm_pir_program_and_param_map()

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            core._set_prim_forward_enabled(True)
            core._set_prim_backward_enabled(True)

            newir_ops = newir_program.global_block().ops
            decompose_bwd_ops_names = [
                "pd_op.layer_norm_grad",
                "pd_op.add_grad",
                "pd_op.multiply_grad",
            ]

            global_outputs = [newir_ops[7].result(0)]
            global_grads = [
                newir_ops[-1].result(0),
                newir_ops[-1].result(1),
                newir_ops[-2].result(1),
            ]

            # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
            (
                related_fwd_ops,
                related_fwd_ops_output_indexes,
            ) = get_graph_outputs_infos(
                newir_program.global_block(), global_outputs
            )  # without update during execution
            (
                related_bwd_ops,
                related_bwd_ops_output_indexes,
            ) = get_graph_outputs_infos(
                newir_program.global_block(), global_grads
            )

            for bwd_op in newir_ops:
                if (
                    flag == "decompose"
                    and bwd_op.name() in decompose_bwd_ops_names
                ):
                    fwd_op = get_fwd_op(bwd_op, grad_var_to_var_map)
                    fwd_inputs = [x.source() for x in fwd_op.operands()]
                    fwd_outputs = fwd_op.results()

                    # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                    if core.has_custom_vjp(fwd_op):
                        related_bwd_op_index = related_graph_outputs(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            newir_program.global_block(),
                            bwd_op,
                            grad_var_to_var_map,
                            fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_graph_outputs(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

                        related_fwd_op_index = related_graph_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            newir_program.global_block(),
                            fwd_op,
                            grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_graph_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                    # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                    else:
                        related_fwd_op_index = related_graph_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            newir_program.global_block(),
                            fwd_op,
                            grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_graph_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                        related_bwd_op_index = related_graph_outputs(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            newir_program.global_block(),
                            bwd_op,
                            grad_var_to_var_map,
                            new_fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_graph_outputs(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

            # execution
            print("final graph: ", newir_program)
            exe = paddle.static.Executor()
            outs = exe.run(
                newir_program,
                feed={'x': self.x, 'y': self.y, 'z': self.z},
                fetch_list=[
                    global_outputs[0],
                    global_grads[0],
                    global_grads[1],
                    global_grads[2],
                ],
            )
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

        return outs

    def test_decompose_layer_norm_op(self):
        res = self.layer_norm_net("decompose")


if __name__ == "__main__":
    unittest.main()
