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
import paddle.tensor
from paddle.autograd.ir_backward import (
    decompose_bwd_op,
    get_global_grads_infos,
    grad,
    related_global_grads,
    replace_global_grads,
)
from paddle.base import core
from paddle.decomposition.decomp import (
    decompose_fwd_op,
    get_global_outputs_infos,
    related_global_outputs,
    replace_global_outputs,
)

paddle.enable_static()


class TestDecomposeOp1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [2, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [2, 3]
        self.z = np.random.random(self.shape_z).astype("float32")

    def layer_norm_net(self, flag=None):
        mp = paddle.pir.core.Program()
        sp = paddle.pir.core.Program()

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(mp, sp):
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

            # construct graph
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            y.stop_gradient = False
            z = paddle.static.data('z', self.shape_z, dtype='float32')
            z.stop_gradient = False
            tmp1 = paddle.add(x, y)
            tmp2 = paddle.multiply(tmp1, z)
            scale = paddle.tensor.fill_constant(
                shape=tmp2.shape[1:],
                dtype=tmp2.dtype,
                value=1.0,
            )
            scale.stop_gradient = False
            bias = paddle.tensor.fill_constant(
                shape=tmp2.shape[1:],
                dtype=tmp2.dtype,
                value=2.0,
            )
            bias.stop_gradient = False
            out = paddle.nn.functional.layer_norm(
                tmp2, tmp2.shape[1:], scale, bias, 1e-5
            )

            # trace forward output and backward grad
            global_outputs = [out]
            global_grads = grad(out, [x, y, z])

            # analysis the graph, and get op mapping infos and variable mapping infos
            ops = mp.global_block().ops
            simu_bwd_op_to_fwd_op_map = {  # without update during execution
                ops[12]: ops[7],
                ops[13]: ops[4],
                ops[14]: ops[3],
            }
            simu_grad_var_to_var_map = {  # with update during execution
                ops[12].operand(5).source(): ops[7].result(0),
                ops[12].result(0): ops[7].operand(0).source(),
                ops[12].result(1): ops[7].operand(1).source(),
                ops[12].result(2): ops[7].operand(2).source(),
                ops[13].result(0): ops[4].operand(0).source(),
                ops[13].result(1): ops[4].operand(1).source(),
                ops[14].result(0): ops[3].operand(0).source(),
                ops[14].result(1): ops[3].operand(1).source(),
            }

            # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
            (
                related_fwd_ops,
                related_fwd_ops_output_indexes,
            ) = get_global_outputs_infos(
                mp.global_block(), global_outputs
            )  # without update during execution
            (
                related_bwd_ops,
                related_bwd_ops_output_indexes,
            ) = get_global_grads_infos(mp.global_block(), global_grads)

            # setting "decompose" flag means decompsing composite op into primitive ops
            if flag == "decompose":
                core._set_prim_forward_enabled(True)
                core._set_prim_backward_enabled(True)
                for bwd_op in simu_bwd_op_to_fwd_op_map:
                    fwd_op = simu_bwd_op_to_fwd_op_map[bwd_op]
                    fwd_inputs = [x.source() for x in fwd_op.operands()]
                    fwd_outputs = fwd_op.results()

                    # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                    if core.has_custom_vjp(fwd_op):
                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )
                    # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                    else:
                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            new_fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

            # execution
            print("final graph: ", mp)
            exe = paddle.static.Executor()
            outs = exe.run(
                mp,
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
        res_ref = self.layer_norm_net()
        res = self.layer_norm_net("decompose")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, atol=1e-4)


class TestDecomposeOp2(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [2, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [2, 3]
        self.z = np.random.random(self.shape_z).astype("float32")

    def gelu_net(self, flag=None):
        mp = paddle.pir.core.Program()
        sp = paddle.pir.core.Program()

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(mp, sp):
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

            # construct graph
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            y.stop_gradient = False
            z = paddle.static.data('z', self.shape_z, dtype='float32')
            z.stop_gradient = False
            tmp1 = paddle.add(x, y)
            tmp2 = paddle.multiply(tmp1, z)
            out = paddle.nn.functional.gelu(tmp2, approximate=False)

            # trace forward output and backward grad
            global_outputs = [out]
            global_grads = grad(out, [x, y, z])

            # analysis the graph, and get op mapping infos and variable mapping infos
            ops = mp.global_block().ops
            simu_bwd_op_to_fwd_op_map = {  # without update during execution
                ops[8]: ops[5],
                ops[9]: ops[4],
                ops[10]: ops[3],
            }
            simu_grad_var_to_var_map = {  # with update during execution
                ops[8].operand(1).source(): ops[5].result(0),
                ops[8].result(0): ops[5].operand(0).source(),
                ops[9].result(0): ops[4].operand(0).source(),
                ops[9].result(1): ops[4].operand(1).source(),
                ops[10].result(0): ops[3].operand(0).source(),
                ops[10].result(1): ops[3].operand(1).source(),
            }

            # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
            (
                related_fwd_ops,
                related_fwd_ops_output_indexes,
            ) = get_global_outputs_infos(
                mp.global_block(), global_outputs
            )  # without update during execution
            (
                related_bwd_ops,
                related_bwd_ops_output_indexes,
            ) = get_global_grads_infos(mp.global_block(), global_grads)

            # setting "decompose" flag means decompsing composite op into primitive ops
            if flag == "decompose":
                core._set_prim_forward_enabled(True)
                core._set_prim_backward_enabled(True)
                for bwd_op in simu_bwd_op_to_fwd_op_map:
                    fwd_op = simu_bwd_op_to_fwd_op_map[bwd_op]
                    fwd_inputs = [x.source() for x in fwd_op.operands()]
                    fwd_outputs = fwd_op.results()

                    # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                    if core.has_custom_vjp(fwd_op):
                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )
                    # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                    else:
                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            new_fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

            # execution
            print("final graph: ", mp)
            exe = paddle.static.Executor()
            outs = exe.run(
                mp,
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
        res_ref = self.gelu_net()
        res = self.gelu_net("decompose")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, atol=1e-4)


class TestDecomposeOp3(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 3]
        self.x = np.random.random(self.shape_x).astype("float32")

    def dropout_net(self, flag=None):
        mp = paddle.pir.core.Program()
        sp = paddle.pir.core.Program()

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(mp, sp):
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

            # construct graph
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            x.stop_gradient = False
            out = paddle.nn.functional.dropout(x, p=0.5)

            # trace forward output and backward grad
            global_outputs = [out]
            global_grads = grad(out, [x])

            # parse the graph, and get op mapping infos and variable mapping infos
            ops = mp.global_block().ops
            simu_bwd_op_to_fwd_op_map = {  # without update during execution
                ops[5]: ops[1],
            }
            simu_grad_var_to_var_map = {  # with update during execution
                ops[5].operand(1).source(): ops[1].result(0),
                ops[5].result(0): ops[1].operand(0).source(),
            }

            # get global outputs and grads info, when decomposing an op that corresponds to global outputs and grads, then update the global outputs and grads
            (
                related_fwd_ops,
                related_fwd_ops_output_indexes,
            ) = get_global_outputs_infos(
                mp.global_block(), global_outputs
            )  # without update during execution
            (
                related_bwd_ops,
                related_bwd_ops_output_indexes,
            ) = get_global_grads_infos(mp.global_block(), global_grads)

            # setting "decompose" flag means decompsing composite op into primitive ops
            if flag == "decompose":
                core._set_prim_forward_enabled(True)
                core._set_prim_backward_enabled(True)
                for bwd_op in simu_bwd_op_to_fwd_op_map:
                    fwd_op = simu_bwd_op_to_fwd_op_map[bwd_op]
                    fwd_inputs = [x.source() for x in fwd_op.operands()]
                    fwd_outputs = fwd_op.results()

                    # if bwd_op has custom_vjp rule, then decompose bwd_op firstly and decompose fwd_op secondly
                    if core.has_custom_vjp(fwd_op):
                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )
                    # if bwd_op has no custom_vjp rule, then decompose fwd_op into a set of primitive ops firstly and decompose bwd_op secondly
                    else:
                        related_fwd_op_index = related_global_outputs(
                            global_outputs, related_fwd_ops, fwd_op
                        )
                        new_fwd_outputs = decompose_fwd_op(
                            mp.global_block(),
                            fwd_op,
                            simu_grad_var_to_var_map,
                        )
                        if related_fwd_op_index is not None:
                            replace_global_outputs(
                                global_outputs,
                                new_fwd_outputs,
                                related_fwd_op_index,
                                related_fwd_ops_output_indexes,
                            )

                        related_bwd_op_index = related_global_grads(
                            global_grads, related_bwd_ops, bwd_op
                        )
                        new_grads = decompose_bwd_op(
                            mp.global_block(),
                            bwd_op,
                            simu_grad_var_to_var_map,
                            new_fwd_outputs,
                            fwd_inputs,
                        )
                        if related_bwd_op_index is not None:
                            replace_global_grads(
                                global_grads,
                                new_grads,
                                related_bwd_op_index,
                                related_bwd_ops_output_indexes,
                            )

            # execution
            print("final graph: ", mp)
            exe = paddle.static.Executor()
            outs = exe.run(
                mp,
                feed={'x': self.x},
                fetch_list=[
                    global_outputs[0],
                    global_grads[0],
                ],
            )

        core._set_prim_backward_enabled(False)
        core._set_prim_forward_enabled(False)
        return outs

    def test_decompose_dropout_op(self):
        res_ref = self.dropout_net()
        res = self.dropout_net("decompose")
        # for ref, actual in zip(res_ref, res):
        #     np.testing.assert_allclose(ref, actual, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
