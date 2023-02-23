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
import pdb


def generate_data(shape, dtype="float32"):
    np_data = np.random.random(shape).astype(dtype)
    return np_data


class Attr:
    def __init__(self) -> None:
        self.dtype = None
        self.shape = None
        self.value = 0.0

    def set_dtype(self, dtype) -> None:
        self.dtype = dtype
        return

    def set_value(self, val) -> None:
        self.value = val
        return

    def set_shape(self, shape) -> None:
        self.shape = shape
        return

    def get_rtol(self, flag):
        rtol = TOLERANCE[self.dtype][flag].get("rtol")
        return rtol

    def get_atol(self, flag):
        atol = TOLERANCE[self.dtype][flag].get("atol")
        return atol


attrs = Attr()


def fn(x):
    return paddle.full_like(x, fill_value = attrs.value, dtype = attrs.dtype)


def expect_forward(inputs):
    return fn(inputs)


class TestCompositeFullLike(unittest.TestCase):
    def setUp(self):
        self.dtypes = ["float16", "float32", "float64"]
        self.shapes = [[2], [2, 3, 4], [2, 3]]
        self.fill_vals = [1.0, 0, -1.0]

    def cal_composite(self, inputs):
        paddle.enable_static()
        core._set_prim_forward_enabled(True)
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                'x', shape=inputs.shape, dtype=str(inputs.dtype)
            )
            y = fn(x)
            blocks = main_program.blocks

            fwd_ops = [op.type for op in blocks[0].ops]
            # Ensure that fill_any_like in original block
            self.assertTrue('fill_any_like' in fwd_ops)

            paddle.incubate.autograd.to_prim(blocks)

            fwd_ops_new = [op.type for op in blocks[0].ops]
            # Ensure that fill_any_like is splitted into small ops
            self.assertTrue('fill_any_like' not in fwd_ops_new)

        exe = paddle.static.Executor()
        exe.run(startup_program)
        res = exe.run(main_program, feed={'x': inputs}, fetch_list=[y])
        paddle.disable_static()
        core._set_prim_forward_enabled(False)
        return res

    def compare_forward(self):
        if not attrs.shape:
            # op softmax does not support both case
            return
        np_data = generate_data(attrs.shape)
        tensor_data = paddle.to_tensor(np_data)

        expect = expect_forward(tensor_data).numpy()
        actual = self.cal_composite(np_data)[0]

        assert expect.dtype == actual.dtype
        np.testing.assert_allclose(
            expect,
            actual,
            rtol=attrs.get_rtol("forward"),
            atol=attrs.get_atol("forward"),
        )

    def test_forward(self):
        for i in self.fill_vals:
            for j in self.dtypes:
                for t in self.shapes:
                    attrs.set_value(i)
                    attrs.set_dtype(j)
                    attrs.set_shape(t)
                    self.compare_forward()


if __name__ == '__main__':
    unittest.main()
