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


def get_pir_program_and_param_map():
    with paddle.pir_utils.OldIrGuard():
        shape = [3, 3]
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
            tmp3 = paddle.matmul(tmp2, z)
            tmp4 = paddle.mean(tmp3, axis=-1, keepdim=True)
            tmp5 = paddle.rsqrt(tmp4)
            scale = paddle.tensor.fill_constant(
                shape=tmp5.shape[1:],
                dtype=tmp5.dtype,
                value=1.0,
            )
            scale.stop_gradient = True
            tmp6 = paddle.nn.functional.layer_norm(
                tmp5, tmp5.shape[1:], scale, None, 1e-5
            )
            tmp7 = paddle.nn.functional.dropout(tmp6, p=0.5)
            tmp8 = paddle.add(x, tmp7)
            tmp9 = paddle.concat(tmp8)

            test = paddle.rand([5, 1, 10])
            tmp_test_1 = paddle.squeeze(test, axis=1)
            out = paddle.mean(tmp9)
            # construct backward graph
            gradients = paddle.static.gradients(out, [x, y, z])

        pir_program, param_mapping = pir.translate_to_pir_with_param_map(
            mp.desc
        )
        return pir_program, param_mapping


class TestDecomposeOp(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [3, 3]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.shape_y = [3, 3]
        self.y = np.random.random(self.shape_y).astype("float32")
        self.shape_z = [3, 3]
        self.z = np.random.random(self.shape_z).astype("float32")

    def net(self, flag=None):
        (
            pir_program,
            param_mapping,
        ) = get_pir_program_and_param_map()

        pir_ops = pir_program.global_block().ops
        fetch_list = [pir_ops[12].result(0)]

        if flag == "decompose":
            core._set_prim_forward_enabled(True)
            core._set_prim_backward_enabled(True)

            # get the grad_var_to_var
            grad_var_to_var = {
                'concat_0.tmp_0@GRAD': 'concat_0.tmp_0',
                'dropout_0.tmp_0@GRAD': 'dropout_0.tmp_0',
                'elementwise_add_0@GRAD': 'elementwise_add_0',
                'elementwise_add_1@GRAD': 'elementwise_add_1',
                'elementwise_mul_0@GRAD': 'elementwise_mul_0',
                'layer_norm_0.tmp_2@GRAD': 'layer_norm_0.tmp_2',
                'matmul_v2_0.tmp_0@GRAD': 'matmul_v2_0.tmp_0',
                'mean_0.tmp_0@GRAD': 'mean_0.tmp_0',
                'mean_1.tmp_0@GRAD': 'mean_1.tmp_0',
                'rsqrt_0.tmp_0@GRAD': 'rsqrt_0.tmp_0',
                'x@GRAD': 'x',
                'x@GRAD@RENAME@block0@0': 'x',
                'x@GRAD@RENAME@block0@1': 'x',
                'y@GRAD': 'y',
                'z@GRAD': 'z',
                'z@GRAD@RENAME@block0@0': 'z',
                'z@GRAD@RENAME@block0@1': 'z',
            }
            decomp.decompose_pir_program(
                pir_program, param_mapping, grad_var_to_var
            )

        with paddle.pir_utils.IrGuard(), paddle.pir.core.program_guard(
            pir_program
        ):
            exe = paddle.static.Executor()
            outs = exe.run(
                pir_program,
                feed={'x': self.x, 'y': self.y, 'z': self.z},
                fetch_list=fetch_list,
            )
            core._set_prim_backward_enabled(False)
            core._set_prim_forward_enabled(False)

        return outs

    def test_decompose_op(self):
        res_ref = self.net()
        res = self.net("decompose")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
