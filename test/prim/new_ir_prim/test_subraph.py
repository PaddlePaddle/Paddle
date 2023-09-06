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

os.environ["FLAGS_enable_new_ir_api"] = "true"
import numpy as np

import paddle

paddle.device.set_device("cpu")
from paddle import nn
from paddle._ir_ops import add, multiply
from paddle.autograd.ir_backward import grad
from paddle.decomposition import decompose
from paddle.framework import core

paddle.enable_static()


def gelu_composite(x, approximate):
    """define composite rule of op gelu"""
    M_SQRT1_2 = (
        0.70710678118654752440  # /* 1/sqrt(2) */ copy from gelu-kernel.cc
    )

    # gelu(x) = 0.5 * x *  (1 + erf(x / sqrt(2)))
    cdf = add(x, paddle.tensor.full(x.shape, M_SQRT1_2, x.dtype))
    # out = x * cdf
    out = multiply(x, cdf)
    return out


class SimpNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            4, epsilon=1e-5, weight_attr=False, bias_attr=False
        )
        self.linear1 = nn.Linear(4, 6)
        self.gelu = nn.GELU(approximate=True)
        self.linear2 = nn.Linear(6, 4)
        self.dropout = nn.Dropout(0.1, mode="upscale_in_train")

    def forward(self, x, weight, bias, linear1_weight, linear2_weight):
        # res = paddle.mean(x, axis=0)
        # x1 = paddle.nn.functional.layer_norm(
        #     x, x.shape[2:], weight=weight, bias=bias, epsilon=1e-05
        # )
        # x2 = _ir_ops.matmul(x, linear1_weight, False, False)
        # x3 = _ir_ops.gelu(x, False)
        x3 = gelu_composite(x, False)
        # breakpoint()
        # x4 = paddle.tensor.matmul(x3, linear2_weight)
        # breakpoint()
        # x1 = self.layer_norm(x)
        # x2 = self.linear1(x1)
        # x3 = self.gelu(x2)
        # x4 = self.linear2(x3)
        # res = self.dropout(x4)
        return x3


class TestPrimMode(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 4, 4]
        self.shape_y = [2, 4, 4]
        self.shape_w = [4]
        self.shape_b = [4]
        self.shape_l1_w = [4, 4, 6]
        self.shape_l2_w = [4, 6, 4]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")
        self.w = np.random.random(self.shape_w).astype("float32")
        self.b = np.random.random(self.shape_b).astype("float32")
        self.l1_w = np.random.random(self.shape_l1_w).astype("float32")
        self.l2_w = np.random.random(self.shape_l2_w).astype("float32")

    def base_net(self, flag=None):
        if flag == "forward":
            core._set_prim_forward_enabled(True)
        elif flag == "backward":
            core._set_prim_backward_enabled(True)
        elif flag == "all":
            core._set_prim_all_enabled(True)
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program):
            net = SimpNet()
            x = paddle.static.data('x', self.shape_x, dtype='float32')
            y = paddle.static.data('y', self.shape_y, dtype='float32')
            x.stop_gradient = False
            y.stop_gradient = False
            w = paddle.static.data('w', self.shape_w, dtype='float32')
            b = paddle.static.data('b', self.shape_b, dtype='float32')
            l1_w = paddle.static.data('l1_w', self.shape_l1_w, dtype='float32')
            l2_w = paddle.static.data('l2_w', self.shape_l2_w, dtype='float32')
            divide_out = paddle.divide(x, y)
            # sum_out = paddle.mean(divide_out, axis=0)
            res = net(divide_out, w, b, l1_w, l2_w)
            # res = paddle.nn.functional.layer_norm(divide_out, divide_out.shape[2:],weight=w, bias=b, epsilon=1e-05)
            [res2] = decompose(main_program, [res])
            gradients = grad(res2, (x, y))

            print(main_program)

            exe = paddle.static.Executor(place=paddle.CPUPlace())
            # breakpoint()
            outs = exe.run(
                feed={
                    'x': self.x,
                    'y': self.y,
                    'w': self.w,
                    'b': self.b,
                    'l1_w': self.l1_w,
                    'l2_w': self.l2_w,
                },
                fetch_list=[res2, gradients[0], gradients[1]],
            )

        whole_ops = [op.name() for op in main_program.block().ops]
        if flag == "forward":
            core._set_prim_forward_enabled(False)
            # assert 'pd.mean' not in whole_ops and 'pd.divide_grad' in whole_ops
        elif flag == "backward":
            core._set_prim_backward_enabled(False)
            # assert 'pd.mean' in whole_ops and 'pd.divide_grad' not in whole_ops
        elif flag == "all":
            core._set_prim_all_enabled(False)
            # assert (
            #     'pd.mean' not in whole_ops and 'pd.divide_grad' not in whole_ops
            # )
        # else:
        # assert 'pd.mean' in whole_ops and 'pd.divide_grad' in whole_ops
        # breakpoint()
        print(outs)
        return outs

    def test_prim_forward(self):
        res_ref = self.base_net()

        # res = self.base_net("forward")

    #     for ref, actual in zip(res_ref, res):
    #         np.testing.assert_allclose(ref, actual, rtol=1e-6)

    # def test_prim_backward(self):
    #     res_ref = self.base_net()
    #     res = self.base_net("backward")
    #     for ref, actual in zip(res_ref, res):
    #         np.testing.assert_allclose(ref, actual, rtol=1e-6)

    # def test_prim_all(self):
    #     res_ref = self.base_net()
    #     res = self.base_net("all")
    #     for ref, actual in zip(res_ref, res):
    #         np.testing.assert_allclose(ref, actual, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
