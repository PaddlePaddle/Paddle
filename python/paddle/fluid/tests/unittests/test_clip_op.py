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

import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestClipOp(OpTest):
    def setUp(self):
        self.max_relative_error = 0.006
        self.python_api = paddle.clip

        self.inputs = {}
        self.initTestCase()

        self.op_type = "clip"
        self.attrs = {}
        self.attrs['min'] = self.min
        self.attrs['max'] = self.max
        if 'Min' in self.inputs:
            min_v = self.inputs['Min']
        else:
            min_v = self.attrs['min']

        if 'Max' in self.inputs:
            max_v = self.inputs['Max']
        else:
            max_v = self.attrs['max']

        input = np.random.random(self.shape).astype(self.dtype)
        input[np.abs(input - min_v) < self.max_relative_error] = 0.5
        input[np.abs(input - max_v) < self.max_relative_error] = 0.5
        self.inputs['X'] = input
        self.outputs = {'Out': np.clip(self.inputs['X'], min_v, max_v)}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_eager=True)
        paddle.disable_static()

    def test_check_grad_normal(self):
        paddle.enable_static()
        self.check_grad(['X'], 'Out', check_eager=True)
        paddle.disable_static()

    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 10, 10)
        self.max = 0.8
        self.min = 0.3
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.1]).astype(self.dtype)


class TestCase1(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16, 8)
        self.max = 0.7
        self.min = 0.0


class TestCase2(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (8, 16)
        self.max = 1.0
        self.min = 0.0


class TestCase3(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.7
        self.min = 0.2


class TestCase4(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


class TestCase5(TestClipOp):
    def initTestCase(self):
        self.dtype = np.float32
        self.shape = (4, 8, 16)
        self.max = 0.5
        self.min = 0.5


class TestCase6(TestClipOp):
    def initTestCase(self):
        self.dtype == np.float16
        self.shape = (4, 8, 8)
        self.max = 0.7
        self.min = 0.2
        self.inputs['Max'] = np.array([0.8]).astype(self.dtype)
        self.inputs['Min'] = np.array([0.3]).astype(self.dtype)


class TestClipOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            input_data = np.random.random((2, 4)).astype("float32")

            def test_Variable():
                fluid.layers.clip(x=input_data, min=-1.0, max=1.0)

            self.assertRaises(TypeError, test_Variable)

            def test_dtype():
                x2 = fluid.layers.data(name='x2', shape=[1], dtype='int32')
                fluid.layers.clip(x=x2, min=-1.0, max=1.0)

            self.assertRaises(TypeError, test_dtype)
        paddle.disable_static()


class TestClipAPI(unittest.TestCase):
    def _executed_api(self, x, min=None, max=None):
        return paddle.clip(x, min, max)

    def test_clip(self):
        paddle.enable_static()
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = fluid.data(name='image', shape=data_shape, dtype='float32')
        min = fluid.data(name='min', shape=[1], dtype='float32')
        max = fluid.data(name='max', shape=[1], dtype='float32')

        place = (
            fluid.CUDAPlace(0)
            if fluid.core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        exe = fluid.Executor(place)

        out_1 = self._executed_api(images, min=min, max=max)
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        out_3 = self._executed_api(images, min=0.3)
        out_4 = self._executed_api(images, max=0.7)
        out_5 = self._executed_api(images, min=min)
        out_6 = self._executed_api(images, max=max)
        out_7 = self._executed_api(images, max=-1.0)
        out_8 = self._executed_api(images)
        out_9 = self._executed_api(
            paddle.cast(images, 'float64'), min=0.2, max=0.9
        )
        out_10 = self._executed_api(
            paddle.cast(images * 10, 'int32'), min=2, max=8
        )
        out_11 = self._executed_api(
            paddle.cast(images * 10, 'int64'), min=2, max=8
        )

        (
            res1,
            res2,
            res3,
            res4,
            res5,
            res6,
            res7,
            res8,
            res9,
            res10,
            res11,
        ) = exe.run(
            fluid.default_main_program(),
            feed={
                "image": data,
                "min": np.array([0.2]).astype('float32'),
                "max": np.array([0.8]).astype('float32'),
            },
            fetch_list=[
                out_1,
                out_2,
                out_3,
                out_4,
                out_5,
                out_6,
                out_7,
                out_8,
                out_9,
                out_10,
                out_11,
            ],
        )

        np.testing.assert_allclose(res1, data.clip(0.2, 0.8), rtol=1e-05)
        np.testing.assert_allclose(res2, data.clip(0.2, 0.9), rtol=1e-05)
        np.testing.assert_allclose(res3, data.clip(min=0.3), rtol=1e-05)
        np.testing.assert_allclose(res4, data.clip(max=0.7), rtol=1e-05)
        np.testing.assert_allclose(res5, data.clip(min=0.2), rtol=1e-05)
        np.testing.assert_allclose(res6, data.clip(max=0.8), rtol=1e-05)
        np.testing.assert_allclose(res7, data.clip(max=-1), rtol=1e-05)
        np.testing.assert_allclose(res8, data, rtol=1e-05)
        np.testing.assert_allclose(
            res9, data.astype(np.float64).clip(0.2, 0.9), rtol=1e-05
        )
        np.testing.assert_allclose(
            res10, (data * 10).astype(np.int32).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            res11, (data * 10).astype(np.int64).clip(2, 8), rtol=1e-05
        )
        paddle.disable_static()

    def test_clip_dygraph(self):
        paddle.disable_static()
        place = (
            fluid.CUDAPlace(0)
            if fluid.core.is_compiled_with_cuda()
            else fluid.CPUPlace()
        )
        paddle.disable_static(place)
        data_shape = [1, 9, 9, 4]
        data = np.random.random(data_shape).astype('float32')
        images = paddle.to_tensor(data, dtype='float32')
        v_min = paddle.to_tensor(np.array([0.2], dtype=np.float32))
        v_max = paddle.to_tensor(np.array([0.8], dtype=np.float32))

        out_1 = self._executed_api(images, min=0.2, max=0.8)
        images = paddle.to_tensor(data, dtype='float32')
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        images = paddle.to_tensor(data, dtype='float32')
        out_3 = self._executed_api(images, min=v_min, max=v_max)

        out_4 = self._executed_api(
            paddle.cast(images * 10, 'int32'), min=2, max=8
        )
        out_5 = self._executed_api(
            paddle.cast(images * 10, 'int64'), min=2, max=8
        )
        # test with numpy.generic
        out_6 = self._executed_api(images, min=np.abs(0.2), max=np.abs(0.8))

        np.testing.assert_allclose(
            out_1.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_2.numpy(), data.clip(0.2, 0.9), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_3.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_4.numpy(), (data * 10).astype(np.int32).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_5.numpy(), (data * 10).astype(np.int64).clip(2, 8), rtol=1e-05
        )
        np.testing.assert_allclose(
            out_6.numpy(), data.clip(0.2, 0.8), rtol=1e-05
        )

    def test_clip_dygraph_default_max(self):
        paddle.disable_static()
        x_int32 = paddle.to_tensor([1, 2, 3], dtype="int32")
        x_int64 = paddle.to_tensor([1, 2, 3], dtype="int64")
        x_f32 = paddle.to_tensor([1, 2, 3], dtype="float32")
        egr_out1 = paddle.clip(x_int32, min=1)
        egr_out2 = paddle.clip(x_int64, min=1)
        egr_out3 = paddle.clip(x_f32, min=1)
        x_int32 = paddle.to_tensor([1, 2, 3], dtype="int32")
        x_int64 = paddle.to_tensor([1, 2, 3], dtype="int64")
        x_f32 = paddle.to_tensor([1, 2, 3], dtype="float32")
        out1 = paddle.clip(x_int32, min=1)
        out2 = paddle.clip(x_int64, min=1)
        out3 = paddle.clip(x_f32, min=1)
        np.testing.assert_allclose(out1.numpy(), egr_out1.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out2.numpy(), egr_out2.numpy(), rtol=1e-05)
        np.testing.assert_allclose(out3.numpy(), egr_out3.numpy(), rtol=1e-05)

    def test_errors(self):
        paddle.enable_static()
        x1 = fluid.data(name='x1', shape=[1], dtype="int16")
        x2 = fluid.data(name='x2', shape=[1], dtype="int8")
        self.assertRaises(TypeError, paddle.clip, x=x1, min=0.2, max=0.8)
        self.assertRaises(TypeError, paddle.clip, x=x2, min=0.2, max=0.8)
        paddle.disable_static()


class TestInplaceClipAPI(TestClipAPI):
    def _executed_api(self, x, min=None, max=None):
        return x.clip_(min, max)


if __name__ == '__main__':
    unittest.main()
