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
        tmp4 = paddle.rsqrt(tmp3)
        scale = paddle.tensor.fill_constant(
            shape=tmp4.shape[1:],
            dtype=tmp4.dtype,
            value=1.0,
        )
        scale.stop_gradient = True
        tmp5 = paddle.nn.functional.layer_norm(
            tmp4, tmp4.shape[1:], scale, None, 1e-5
        )
        tmp6 = paddle.nn.functional.dropout(tmp5, p=0.5)
        out = paddle.add(x, tmp6)
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
        global_outputs = [newir_ops[10].result(0)]

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            newir_program
        ):
            if flag == "decompose":
                core._set_prim_forward_enabled(True)
                core._set_prim_backward_enabled(True)

                # get the old_ir_grad_var_to_var map
                old_ir_grad_var_to_var_map = {
                    'dropout_1.tmp_0@GRAD': 'dropout_1.tmp_0',
                    'elementwise_add_2@GRAD': 'elementwise_add_2',
                    'elementwise_add_3@GRAD': 'elementwise_add_3',
                    'elementwise_mul_1@GRAD': 'elementwise_mul_1',
                    'layer_norm_1.tmp_2@GRAD': 'layer_norm_1.tmp_2',
                    'rsqrt_1.tmp_0@GRAD': 'rsqrt_1.tmp_0',
                    'mean_1.tmp_0@GRAD': 'mean_1.tmp_0',
                    'x@GRAD': 'x',
                    'x@GRAD@RENAME@block0@0': 'x',
                    'x@GRAD@RENAME@block0@1': 'x',
                    'y@GRAD': 'y',
                    'z@GRAD': 'z',
                }
                grad_var_to_var_map = get_new_ir_grad_var_to_var_map(
                    param_mappings, old_ir_grad_var_to_var_map
                )
                bwd_ops_to_be_decomposed = [
                    "pd_op.layer_norm_grad",
                    "pd_op.dropout_grad",
                    "pd_op.mean_grad",
                    "pd_op.add_grad",
                    "pd_op.multiply_grad",
                    "pd_op.rsqrt_grad",
                ]
                for bwd_op in newir_ops:
                    if (
                        flag == "decompose"
                        and bwd_op.name() in bwd_ops_to_be_decomposed
                    ):
                        new_grads, has_decomposed = decomp.decomp_bwd_op(
                            newir_program.global_block(),
                            bwd_op,
                            grad_var_to_var_map,
                        )

            # execution
            exe = paddle.static.Executor()
            outs = exe.run(
                newir_program,
                feed={'x': self.x, 'y': self.y, 'z': self.z},
                fetch_list=[
                    global_outputs[0],
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
