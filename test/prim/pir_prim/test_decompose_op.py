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


import unittest

import numpy as np

import paddle
from paddle import pir
from paddle.base import core
from paddle.decomposition import decomp

paddle.enable_static()


def check_param_mappings(param_mappings):
    for VarDesc, Values in param_mappings.items():
        if len(Values) < 0 or len(Values) > 1:
            raise ValueError("currently only support one-to-one param_mappings")


def get_new_ir_grad_var_to_var_map(param_mappings, old_ir_grad_var_to_var_map):
    new_ir_grad_var_to_var_map = {}
    for grad_var, var in old_ir_grad_var_to_var_map.items():
        if grad_var in param_mappings.keys():
            new_grad_var = param_mappings[grad_var][0]
            new_var = param_mappings[var][0]
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


def get_pir_program_and_param_map():
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
        tmp3 = paddle.mean(tmp2, axis=-1, keepdim=True)
        scale = paddle.tensor.fill_constant(
            shape=tmp3.shape[1:],
            dtype=tmp3.dtype,
            value=1.0,
        )
        scale.stop_gradient = False
        out = paddle.nn.functional.layer_norm(
            tmp3, tmp3.shape[1:], scale, None, 1e-5
        )
        # construct backward graph
        gradients = paddle.static.gradients(out, [x, y, z])

    newir_program, param_mappings = pir.translate_to_new_ir_with_param_map(
        mp.desc
    )
    check_param_mappings(param_mappings)

    return newir_program, param_mappings


class TestDecomposeOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [2, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [2, 3]
        self.z = np.random.random(self.shape_z).astype("float32")

    def net(self, flag=None):
        (
            newir_program,
            param_mappings,
        ) = get_pir_program_and_param_map()

        newir_ops = newir_program.global_block().ops
        global_outputs = [newir_ops[7].result(0)]
        global_grads = [
            newir_ops[-1].result(0),
            newir_ops[-1].result(1),
            newir_ops[-2].result(1),
        ]

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            if flag == "decompose":
                core._set_prim_forward_enabled(True)
                core._set_prim_backward_enabled(True)

                # get the old_ir_grad_var_to_var map
                old_ir_grad_var_to_var_map = {
                    'layer_norm_1.tmp_2@GRAD': 'layer_norm_1.tmp_2',
                    'mean_1.tmp_0@GRAD': 'mean_1.tmp_0',
                    "fill_constant_3.tmp_0@GRAD": "fill_constant_3.tmp_0",
                    'elementwise_mul_1@GRAD': 'elementwise_mul_1',
                    'elementwise_add_1@GRAD': 'elementwise_add_1',
                    'z@GRAD': 'z',
                    'x@GRAD': 'x',
                    'y@GRAD': 'y',
                }

                grad_var_to_var_map = get_new_ir_grad_var_to_var_map(
                    param_mappings, old_ir_grad_var_to_var_map
                )

                # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
                (
                    fwd_leaf_ops,
                    fwd_leaf_ops_output_indexes,
                ) = decomp.get_leaf_ops(
                    newir_program.global_block(), global_outputs
                )  # without update during execution
                (
                    bwd_leaf_ops,
                    bwd_leaf_ops_output_indexes,
                ) = decomp.get_leaf_ops(
                    newir_program.global_block(), global_grads
                )

                decompose_bwd_ops_names = [
                    "pd_op.layer_norm_grad",
                    "pd_op.mean_grad",
                    "pd_op.add_grad",
                    "pd_op.multiply_grad",
                ]
                for bwd_op in newir_ops:
                    if (
                        flag == "decompose"
                        and bwd_op.name() in decompose_bwd_ops_names
                    ):
                        fwd_op = get_fwd_op(bwd_op, grad_var_to_var_map)
                        assert fwd_op is not None, "fwd_op is None"
                        fwd_inputs = tuple(
                            x.source() for x in fwd_op.operands()
                        )
                        fwd_outputs = tuple(fwd_op.results())

                        # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                        if core.has_custom_vjp(fwd_op):
                            bwd_leaf_op_index = (
                                bwd_leaf_ops.index(bwd_op)
                                if bwd_op in bwd_leaf_ops
                                else None
                            )
                            new_grads = decomp.decompose_bwd_op(
                                newir_program.global_block(),
                                bwd_op,
                                grad_var_to_var_map,
                                fwd_outputs,
                                fwd_inputs,
                            )
                            if bwd_leaf_op_index is not None:
                                decomp.replace_graph_outputs(
                                    global_grads,
                                    new_grads,
                                    bwd_leaf_op_index,
                                    bwd_leaf_ops_output_indexes,
                                )

                            fwd_leaf_op_index = (
                                fwd_leaf_ops.index(fwd_op)
                                if fwd_op in fwd_leaf_ops
                                else None
                            )
                            new_fwd_outputs = decomp.decompose_fwd_op(
                                newir_program.global_block(),
                                fwd_op,
                                grad_var_to_var_map,
                            )
                            if fwd_leaf_op_index is not None:
                                decomp.replace_graph_outputs(
                                    global_outputs,
                                    new_fwd_outputs,
                                    fwd_leaf_op_index,
                                    fwd_leaf_ops_output_indexes,
                                )

                        # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                        else:
                            fwd_leaf_op_index = (
                                fwd_leaf_ops.index(fwd_op)
                                if fwd_op in fwd_leaf_ops
                                else None
                            )
                            new_fwd_outputs = decomp.decompose_fwd_op(
                                newir_program.global_block(),
                                fwd_op,
                                grad_var_to_var_map,
                            )
                            if fwd_leaf_op_index is not None:
                                decomp.replace_graph_outputs(
                                    global_outputs,
                                    new_fwd_outputs,
                                    fwd_leaf_op_index,
                                    fwd_leaf_ops_output_indexes,
                                )

                            bwd_leaf_op_index = (
                                bwd_leaf_ops.index(bwd_op)
                                if bwd_op in bwd_leaf_ops
                                else None
                            )
                            new_grads = decomp.decompose_bwd_op(
                                newir_program.global_block(),
                                bwd_op,
                                grad_var_to_var_map,
                                new_fwd_outputs,
                                fwd_inputs,
                            )
                            if bwd_leaf_op_index is not None:
                                decomp.replace_graph_outputs(
                                    global_grads,
                                    new_grads,
                                    bwd_leaf_op_index,
                                    bwd_leaf_ops_output_indexes,
                                )

            # execution
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
        res_ref = self.net()
        res = self.net("decompose")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
