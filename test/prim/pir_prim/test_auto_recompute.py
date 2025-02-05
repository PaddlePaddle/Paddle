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

import os
import unittest

import numpy as np
import parameterized as param

import paddle
from paddle.autograd.ir_backward import grad as ir_grad
from paddle.base import core
from paddle.decomposition import decompose

TOLERANCE = {
    "float64": {"rtol": 1e-15, "atol": 1e-15},
    "float32": {"rtol": 1e-6, "atol": 1e-6},
    "float16": {"rtol": 1e-3, "atol": 1e-3},
    "bfloat16": {"rtol": 1e-2, "atol": 1e-2},
}


def rms_norm(weight, hidden):
    variance = paddle.mean(paddle.pow(hidden, 2), axis=-1, keepdim=True)
    hidden = paddle.rsqrt(variance + 0.00001) * hidden
    return hidden * weight


places = []
if (
    os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
    in ['1', 'true', 'on']
    or not paddle.is_compiled_with_cuda()
):
    places.append(paddle.CPUPlace())
if paddle.is_compiled_with_cuda():
    places.append(paddle.CUDAPlace(0))


@param.parameterized_class(
    ('name', 'inputs', 'dtype', 'places'),
    (
        (
            "auto_recompute_rms_norm_test1",
            [
                np.random.random(size=[4096, 4096]),
                np.random.random(size=[4096, 4096]),
            ],
            "float32",
            places,
        ),
        (
            "auto_recompute_rms_norm_test2",
            [
                np.random.random(size=[128, 256]),
                np.random.random(size=[128, 256]),
            ],
            "float32",
            places,
        ),
    ),
)
class TestAutoRecomputeRmsNorm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = [
            (
                x.astype(cls.dtype)
                if cls.dtype != "bfloat16"
                else x.astype("float32")
            )
            for x in cls.inputs
        ]
        core._set_prim_all_enabled(True)
        paddle.enable_static()

    @classmethod
    def tearDownClass(cls):
        core._set_prim_all_enabled(False)
        paddle.disable_static()

    def product_rms_norm_inputs(self):
        weight = paddle.static.data(
            name="weight", shape=self.inputs[0].shape, dtype=self.dtype
        )
        hidden = paddle.static.data(
            name="hidden", shape=self.inputs[1].shape, dtype=self.dtype
        )
        weight.stop_gradient = False
        hidden.stop_gradient = False
        return [weight, hidden]

    def cal_rms_norm_decomp_res(self, place):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            weight, hidden = self.product_rms_norm_inputs()
            out = rms_norm(weight, hidden)
            out_grad = paddle.full(
                shape=out.shape, fill_value=3, dtype="float32"
            )
            [out] = decompose(main_program, [out])
            [dweight, dhidden] = ir_grad(out, [weight, hidden], out_grad)
            exe = paddle.static.Executor(place)
            res = exe.run(
                feed={'weight': self.inputs[0], 'hidden': self.inputs[1]},
                fetch_list=[dweight, dhidden],
            )
        return res, main_program

    def cal_rms_norm_auto_recompute_decomp_res(self, place):
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            weight, hidden = self.product_rms_norm_inputs()
            out = rms_norm(weight, hidden)
            out_grad = paddle.full(
                shape=out.shape, fill_value=3, dtype="float32"
            )
            [out] = decompose(main_program, [out])
            [dweight, dhidden] = ir_grad(out, [weight, hidden], out_grad)
            main_program, _ = paddle.decomposition.auto_recompute(
                main_program,
                [weight, hidden],
                [out],
                grad_outputs=[out_grad],
                fwd_op_end_idx=13,
                backward_op_start_idx=15,
            )
            exe = paddle.static.Executor(place)
            res = exe.run(
                feed={'weight': self.inputs[0], 'hidden': self.inputs[1]},
                fetch_list=[dweight, dhidden],
            )
        return res, main_program

    def test_auto_recompute(self):
        for place in places:
            res_desire, orig_program = self.cal_rms_norm_decomp_res(place)
            (
                res_recompute,
                recompute_program,
            ) = self.cal_rms_norm_auto_recompute_decomp_res(place)
            np.testing.assert_allclose(
                res_desire[0],
                res_recompute[0],
                atol=TOLERANCE[self.dtype]["atol"],
                rtol=TOLERANCE[self.dtype]["rtol"],
            )
            np.testing.assert_allclose(
                res_desire[1],
                res_recompute[1],
                atol=TOLERANCE[self.dtype]["atol"],
                rtol=TOLERANCE[self.dtype]["rtol"],
            )
            forward_ops = recompute_program.global_block().ops[:13]
            backward_ops = recompute_program.global_block().ops[13:]
            saved_values = forward_ops[10].results()[0]
            define_op = saved_values.get_defining_op()
            for op in forward_ops:
                if op.name() == "pd_op.data":
                    continue
                op_results = op.results()
                for op_result in op_results:
                    if op_result.is_same(saved_values):
                        continue
                    else:
                        all_used_ops = op_result.all_used_ops()


if __name__ == '__main__':
    unittest.main()
