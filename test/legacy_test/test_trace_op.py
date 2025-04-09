# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import base, tensor
from paddle.base import core


class TestTraceOp(OpTest):
    def setUp(self):
        self.op_type = "trace"
        self.python_api = paddle.trace
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out', check_pir=True)

    def init_config(self):
        self.case = np.random.randn(20, 6).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.trace(self.inputs['Input'])


class TestTraceOpCase1(TestTraceOp):
    def init_config(self):
        self.case = np.random.randn(2, 20, 2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 1, 'axis1': 0, 'axis2': 2}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'],
        )


class TestTraceOpCase2(TestTraceOp):
    def init_config(self):
        self.case = np.random.randn(2, 20, 2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -5, 'axis1': 1, 'axis2': -1}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'],
        )


class TestTraceFP16Op1(TestTraceOp):
    def init_config(self):
        self.dtype = np.float16
        self.case = np.random.randn(20, 6).astype(self.dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.trace(self.inputs['Input'])


class TestTraceFP16Op2(TestTraceOp):
    def init_config(self):
        self.dtype = np.float16
        self.case = np.random.randn(2, 20, 2, 3).astype(self.dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -5, 'axis1': 1, 'axis2': -1}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'],
        )


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestTraceBF16Op1(OpTest):
    def setUp(self):
        self.op_type = "trace"
        self.python_api = paddle.trace
        self.init_config()
        self.outputs = {'Out': self.target}

        self.inputs['Input'] = convert_float_to_uint16(self.inputs['Input'])
        self.outputs['Out'] = convert_float_to_uint16(self.outputs['Out'])
        self.place = core.CUDAPlace(0)

    def test_check_output(self):
        self.check_output_with_place(self.place, check_pir=True)

    def test_check_grad(self):
        self.check_grad_with_place(
            self.place,
            ['Input'],
            'Out',
            numeric_grad_delta=0.02,
            check_pir=True,
        )

    def init_config(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.case = np.random.randn(20, 6).astype(self.np_dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.trace(self.inputs['Input'])


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestTraceBF16Op2(TestTraceBF16Op1):
    def init_config(self):
        self.dtype = np.uint16
        self.np_dtype = np.float32
        self.case = np.random.randn(2, 20, 2, 3).astype(self.np_dtype)
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -5, 'axis1': 1, 'axis2': -1}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'],
        )


class TestTraceAPICase(unittest.TestCase):

    def test_case1(self):
        with paddle.static.program_guard(paddle.static.Program()):
            case = np.random.randn(2, 20, 2, 3).astype('float32')
            data1 = paddle.static.data(
                name='data1', shape=[2, 20, 2, 3], dtype='float32'
            )
            out1 = tensor.trace(data1)
            out2 = tensor.trace(data1, offset=-5, axis1=1, axis2=-1)

            place = core.CPUPlace()
            exe = base.Executor(place)
            results = exe.run(
                paddle.static.default_main_program(),
                feed={"data1": case},
                fetch_list=[out1, out2],
                return_numpy=True,
            )
        target1 = np.trace(case)
        target2 = np.trace(case, offset=-5, axis1=1, axis2=-1)
        np.testing.assert_allclose(results[0], target1, rtol=1e-05)
        np.testing.assert_allclose(results[1], target2, rtol=1e-05)


class TestTraceAPIZerodimCase(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))
        self.x = np.random.random([5, 0, 0, 0]).astype('float32')

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            x = paddle.to_tensor(self.x, place=place)
            params = [
                (0, 1, 2),
                (1, 0, 1),
                (-1, 2, 0),
                (2, 1, 2),
                (0, -1, -2),
                (5, 1, 2),
                (-5, 2, 0),
            ]
            for offset, axis1, axis2 in params:
                paddle_res = paddle.trace(
                    x, offset=offset, axis1=axis1, axis2=axis2
                )
                np_res = np.trace(
                    self.x, offset=offset, axis1=axis1, axis2=axis2
                )
                self.assertEqual(tuple(paddle_res.shape), np_res.shape)
                np.testing.assert_allclose(paddle_res, np_res, rtol=1e-6)
        paddle.enable_static()

    def test_static(self):
        with paddle.static.program_guard(paddle.static.Program()):
            case = np.random.randn(2, 0, 0, 0).astype('float32')
            data1 = paddle.static.data(
                name='data1', shape=[2, 0, 0, 0], dtype='float32'
            )
            params = [
                (0, 1, 2),
                (-5, 1, -1),
                (2, 0, 1),
                (0, 2, 1),
                (1, 0, 2),
                (-1, 1, 0),
                (0, -2, -1),
            ]
            for offset, axis1, axis2 in params:
                out = tensor.trace(
                    data1, offset=offset, axis1=axis1, axis2=axis2
                )
                place = core.CPUPlace()
                exe = base.Executor(place)
                result = exe.run(
                    paddle.static.default_main_program(),
                    feed={"data1": case},
                    fetch_list=[out],
                    return_numpy=True,
                )[0]
                target = np.trace(case, offset=offset, axis1=axis1, axis2=axis2)
                self.assertEqual(tuple(result.shape), target.shape)
                np.testing.assert_allclose(result, target, rtol=1e-5)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
