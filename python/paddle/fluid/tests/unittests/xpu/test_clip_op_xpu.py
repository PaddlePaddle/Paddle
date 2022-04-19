#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test_xpu import OpTest, XPUOpTest
import paddle
from paddle.fluid import Program, program_guard

import op_test
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


class XPUTestClipOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'clip'
        self.use_dynamic_create_class = False

    class TestClipOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "clip"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_data()
            self.set_attrs()
            self.set_inputs()
            self.outputs = {
                'Out': np.clip(self.inputs['X'], self.min_v, self.max_v)
            }

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def init_data(self):
            self.shape = (4, 10, 10)
            self.max = 0.8
            self.min = 0.3

        def set_inputs(self):
            if 'Min' in self.inputs:
                min_v = self.inputs['Min']
            else:
                min_v = self.attrs['min']

            if 'Max' in self.inputs:
                max_v = self.inputs['Max']
            else:
                max_v = self.attrs['max']

            self.min_v = min_v
            self.max_v = max_v
            self.max_relative_error = 0.006
            input = np.random.random(self.shape).astype("float32")
            input[np.abs(input - min_v) < self.max_relative_error] = 0.5
            input[np.abs(input - max_v) < self.max_relative_error] = 0.5
            self.inputs['X'] = input

        def set_attrs(self):
            self.attrs = {}
            self.attrs['min'] = self.min
            self.attrs['max'] = self.max

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            paddle.enable_static()
            self.check_output_with_place(self.place)
            paddle.disable_static()

    class TestClipOp1(TestClipOp):
        def init_data(self):
            self.shape = (8, 16, 8)
            self.max = 0.7
            self.min = 0.0

    class TestClipOp2(TestClipOp):
        def init_data(self):
            self.shape = (8, 16)
            self.max = 1.0
            self.min = 0.0

    class TestClipOp3(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 16)
            self.max = 0.7
            self.min = 0.2

    class TestClipOp4(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 8)
            self.max = 0.7
            self.min = 0.2
            self.inputs['Max'] = np.array([0.8]).astype('float32')
            self.inputs['Min'] = np.array([0.3]).astype('float32')

    class TestClipOp5(TestClipOp):
        def init_data(self):
            self.shape = (4, 8, 16)
            self.max = 0.5
            self.min = 0.5


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

        place = fluid.XPUPlace(0) if fluid.core.is_compiled_with_xpu(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)

        out_1 = self._executed_api(images, min=min, max=max)
        out_2 = self._executed_api(images, min=0.2, max=0.9)
        out_3 = self._executed_api(images, min=0.3)
        out_4 = self._executed_api(images, max=0.7)
        out_5 = self._executed_api(images, min=min)
        out_6 = self._executed_api(images, max=max)
        out_7 = self._executed_api(images, max=-1.)
        out_8 = self._executed_api(images)

        res1, res2, res3, res4, res5, res6, res7, res8 = exe.run(
            fluid.default_main_program(),
            feed={
                "image": data,
                "min": np.array([0.2]).astype('float32'),
                "max": np.array([0.8]).astype('float32')
            },
            fetch_list=[
                out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8
            ])

        self.assertTrue(np.allclose(res1, data.clip(0.2, 0.8)))
        self.assertTrue(np.allclose(res2, data.clip(0.2, 0.9)))
        self.assertTrue(np.allclose(res3, data.clip(min=0.3)))
        self.assertTrue(np.allclose(res4, data.clip(max=0.7)))
        self.assertTrue(np.allclose(res5, data.clip(min=0.2)))
        self.assertTrue(np.allclose(res6, data.clip(max=0.8)))
        self.assertTrue(np.allclose(res7, data.clip(max=-1)))
        self.assertTrue(np.allclose(res8, data))
        paddle.disable_static()

    def test_clip_dygraph(self):
        paddle.disable_static()
        place = fluid.XPUPlace(0) if fluid.core.is_compiled_with_xpu(
        ) else fluid.CPUPlace()
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

        self.assertTrue(np.allclose(out_1.numpy(), data.clip(0.2, 0.8)))
        self.assertTrue(np.allclose(out_2.numpy(), data.clip(0.2, 0.9)))
        self.assertTrue(np.allclose(out_3.numpy(), data.clip(0.2, 0.8)))

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


support_types = get_xpu_op_support_types('clip')
for stype in support_types:
    create_test_class(globals(), XPUTestClipOp, stype)

if __name__ == '__main__':
    unittest.main()
