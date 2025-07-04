#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op import Operator
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core
from paddle.nn import clip


class TestClipByNormOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = clip.clip_by_norm
        self.init_dtype()
        self.initTestCase()
        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "clip_by_norm"
        self.inputs = {
            'X': input,
        }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > self.max_norm:
            output = self.max_norm * input / norm
        else:
            output = input
        self.outputs = {'Out': output}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def initTestCase(self):
        self.shape = (100,)
        self.max_norm = 1.0

    def init_dtype(self):
        self.dtype = np.float32


class TestCase1(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (100,)
        self.max_norm = 1e20


class TestCase2(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestCase3(TestClipByNormOp):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


class TestClipByNormOpFp16(TestClipByNormOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=0.001, check_pir=True)


class TestClipByNormOpFp16Case1(TestClipByNormOpFp16):
    def initTestCase(self):
        self.shape = (100,)
        self.max_norm = 1e20


class TestClipByNormOpFp16Case2(TestClipByNormOpFp16):
    def initTestCase(self):
        self.shape = (16, 16)
        self.max_norm = 0.1


class TestClipByNormOpFp16Case3(TestClipByNormOpFp16):
    def initTestCase(self):
        self.shape = (4, 8, 16)
        self.max_norm = 1.0


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestClipByNormBF16Op(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = clip.clip_by_norm
        self.init_dtype()
        self.initTestCase()
        input = np.random.random(self.shape).astype(self.np_dtype)
        input[np.abs(input) < self.max_relative_error] = 0.5
        self.op_type = "clip_by_norm"
        self.inputs = {
            'X': input,
        }
        self.attrs = {}
        self.attrs['max_norm'] = self.max_norm
        norm = np.sqrt(np.sum(np.square(input)))
        if norm > self.max_norm:
            output = self.max_norm * input / norm
        else:
            output = input
        self.outputs = {'Out': output}

        self.inputs['X'] = convert_float_to_uint16(self.inputs['X'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def initTestCase(self):
        self.shape = (100,)
        self.max_norm = 1.0

    def init_dtype(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32


class TestClipByNormOpWithSelectedRows(unittest.TestCase):
    def check_with_place(self, place):
        self.config_test_case()
        scope = core.Scope()

        # set input
        x_selected_rows = scope.var('X').get_selected_rows()
        x_selected_rows.set_rows(self.grad_rows)
        x_tensor = x_selected_rows.get_tensor()
        x_np = np.random.random(self.grad_shape).astype("float32")
        x_np[np.abs(x_np) < self.max_relative_error] = 0.5
        x_tensor.set(x_np, place)

        # set output
        out_selected_rows = scope.var('Out').get_selected_rows()

        # run clip_by_norm_op
        clip_by_norm_op = Operator(
            "clip_by_norm", max_norm=self.max_norm, X='X', Out='Out'
        )
        clip_by_norm_op.run(scope, place)

        # check output
        self.assertEqual(out_selected_rows.rows(), self.grad_clipped_rows)
        out_tensor = out_selected_rows.get_tensor()
        y_np = np.zeros(self.grad_clipped_shape)
        y_np[0] = np.sum(x_np[0:2])
        y_np[1] = x_np[2]
        y_np[2] = x_np[3]
        norm = np.sqrt(np.sum(np.square(y_np)))
        if norm > self.max_norm:
            output = self.max_norm * y_np / norm
        else:
            output = y_np
        np.testing.assert_allclose(
            np.array(out_tensor),
            output,
            rtol=1e-05,
            atol=1e-05,
            equal_nan=False,
        )

    def test_clip_by_norm_with_selected_ros(self):
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(core.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            self.check_with_place(place)

    def config_test_case(self):
        self.max_norm = 1.0
        self.max_relative_error = 0.006
        self.grad_shape = (4, 1)
        self.grad_clipped_shape = (3, 1)
        self.grad_rows = [0, 0, 1, 2]
        self.grad_clipped_rows = [0, 1, 2]


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
