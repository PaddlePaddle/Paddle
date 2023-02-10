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
from utils import TOLERANCE

import paddle
import paddle.nn.functional as F
from paddle.fluid import core
from paddle import _C_ops, in_dynamic_mode


def generate_data(shape1, shape2, shape3, dtype="float32"):
    np.random.seed(100)
    np_data1 = np.random.random(shape1).astype(dtype)
    np_data2 = np.random.random(shape2).astype(dtype)
    np_data3 = np.random.random(shape3).astype(dtype)
    return np_data1, np_data2, np_data3


class Attr:
    def __init__(self) -> None:
        self.dtype = None
        self.n_shape = None
        self.shape1 = None
        self.shape2 = None
        self.shape3 = None
       

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype
        return

    def set_shape(self, n_shape, shape1, shape2, shape3) -> None:
        self.n_shape = n_shape
        self.shape1 = shape1
        self.shape2 = shape2
        self.shape3 = shape3
        return

    def get_rtol(self, flag):
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x, norm_shape, w, b):
    return F.layer_norm(x, norm_shape, w, b)

def layer_norm_ (input, weight, bias, epsilon=1e-05, begin_norm_axis = 0):
    axis = np.arange(begin_norm_axis,len(input.shape))
    mean = paddle.mean(input, axis=axis, keepdim=True)
    t1 = input - mean
    t2 = paddle.pow( t1, 2.0)
    t3 = paddle.mean( t2, axis=axis, keepdim=True)
    t4 = t3 + epsilon
    t5 = paddle.sqrt( t4 )
    t7 = t1 / t5
    out = t7
    if weight is not None:
        weight = paddle.reshape(weight, input.shape[begin_norm_axis:])
        out = t7 * paddle.broadcast_to(weight, out.shape)
    if bias is not None:
        bias = paddle.reshape(bias, input.shape[begin_norm_axis:])
        out = out + paddle.broadcast_to(bias, out.shape)
    
    return out

def composite_forward(x, norm_shape, w, b):
    b_axis = len(x.shape) - len(norm_shape)
    return layer_norm_(x, w, b, begin_norm_axis=b_axis)



def expect_forward(x, norm_shape, w, b):
    return fn(x, norm_shape, w, b)


class TestCompositelayer_norm(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float16", "float32"]
        self.n_shape = [[3, 4],[3], [2, 3]]
        self.shape1s = [[3, 4],[2, 4, 3], [2, 2, 3]]
        self.shape2s = [[12],[3],[6]]
        self.shape3s = [[12],[3],[6]]

    def cal_composite(self, inputs, norm_shape, weight, bias):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            w = paddle.static.data(
                'w', shape=weight.shape, dtype=str(weight.dtype)
            )
            b = paddle.static.data(
                'b', shape=bias.shape, dtype=str(bias.dtype)
            )
            y = fn(x, norm_shape, w, b)
            
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm in original block
            self.assertTrue('layer_norm' in fwd_ops)

            paddle.incubate.autograd.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that layer_norm is splitted into small ops
            self.assertTrue('layer_norm' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(
            main_program, 
            feed={
                'x': inputs,
                'w': weight,
                'b': bias,
            }, 
            fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        x, w, b = generate_data(attrs.shape1, attrs.shape2, attrs.shape3)
        n_shape = attrs.n_shape
        x_p = paddle.to_tensor(x)
        w_p = paddle.to_tensor(w)
        b_p = paddle.to_tensor(b)

        expect = expect_forward(x_p, n_shape, w_p, b_p).numpy()
        
        print("expect = ", expect)
        #actual = self.cal_composite(x_p, n_shape, w_p, b_p)
        actual = composite_forward(x_p, n_shape, w_p, b_p).numpy()
        
        print("actual = ", actual)
        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

        expect_2 = expect_forward(x_p, n_shape, None, None).numpy()
        actual_2 = composite_forward(x_p, n_shape, None, None).numpy()
        assert expect_2.dtype == actual_2.dtype
        np.testing.assert_allclose(
            expect_2,
            actual_2,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for j in self.dtypes:
            for t in range(0,len(self.shape1s)):
                attrs.set_dtype(j)
                attrs.set_shape(self.n_shape[t], self.shape1s[t], self.shape2s[t], self.shape3s[t])
                self.compare_forward()


if __name__ == '__main__':
    unittest.main()
