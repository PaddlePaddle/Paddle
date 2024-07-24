#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
from paddle.base import Program, program_guard


class XPUTestScaleOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'scale'
        self.use_dynamic_create_class = False

    class TestScaleOp(XPUOpTest):
        def setUp(self):
            self.init_dtype()
            self.set_xpu()
            self.op_type = "scale"
            self.place = paddle.XPUPlace(0)
            self.set_shape()
            self.set_inputs()
            self.set_attrs()
            self.set_output()

        def set_xpu(self):
            self.__class__.use_xpu = True
            self.__class__.no_need_check_grad = True
            self.__class__.op_type = self.dtype

        def set_inputs(self):
            if self.dtype == np.uint16:
                x = np.random.random(self.shape).astype('float32')
                self.inputs = {'X': convert_float_to_uint16(x)}
            else:
                self.inputs = {
                    'X': np.random.random(self.shape).astype(self.dtype)
                }

        def set_output(self):
            if self.dtype == np.uint16:
                output = (
                    convert_uint16_to_float(self.inputs['X'])
                    * self.attrs['scale']
                )
            else:
                output = self.inputs['X'] * self.attrs['scale']

            self.outputs = {'Out': output}

        def init_dtype(self):
            self.dtype = self.in_type

        def set_attrs(self):
            self.attrs = {'scale': -2.3}

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place)

        def set_shape(self):
            self.shape = [10, 10]

    class TestScaleOp1(TestScaleOp):
        def set_attrs(self):
            self.attrs = {'scale': 3.5}

    class TestScaleOp2(TestScaleOp):
        def set_attrs(self):
            self.attrs = {'scale': 6.77}

    class TestScaleOp3(TestScaleOp):
        def set_attrs(self):
            self.attrs = {'scale': -9.19}

    class TestScaleOp4(TestScaleOp):
        def set_attrs(self):
            self.attrs = {'scale': 0.0}

    class TestScaleOp5(TestScaleOp):
        def set_attrs(self):
            self.attrs = {'scale': -0.003}

    @check_run_big_shape_test()
    class TestScaleOpLargeShape1(TestScaleOp):
        def set_shape(self):
            self.shape = [64]

    @check_run_big_shape_test()
    class TestScaleOpLargeShape2(TestScaleOp):
        def set_shape(self):
            self.shape = [8192, 1]

    @check_run_big_shape_test()
    class TestScaleOpLargeShape3(TestScaleOp):
        def set_shape(self):
            self.shape = [1, 8192, 5, 64]

    @check_run_big_shape_test()
    class TestScaleOpLargeShape4(TestScaleOp):
        def set_shape(self):
            self.shape = [8192, 1920]

    @check_run_big_shape_test()
    class TestScaleOpLargeShape5(TestScaleOp):
        def set_shape(self):
            self.shape = [1024, 5120]

    @check_run_big_shape_test()
    class TestScaleOpLargeShape6(TestScaleOp):
        def set_shape(self):
            self.shape = [8192, 3456]


class TestScaleApiStatic(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.enable_static()
        input = np.random.random([2, 25]).astype("float32")
        main_prog = Program()
        with program_guard(main_prog, Program()):
            x = paddle.static.data(name="x", shape=[2, 25], dtype="float32")
            out = self._executed_api(x, scale=2.0, bias=3.0)

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        out = exe.run(main_prog, feed={"x": input}, fetch_list=[out])
        np.testing.assert_array_equal(out[0], input * 2.0 + 3.0)


class TestScaleInplaceApiStatic(TestScaleApiStatic):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


class TestScaleApiDygraph(unittest.TestCase):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return paddle.scale(x, scale, bias)

    def test_api(self):
        paddle.disable_static()
        input = np.random.random([2, 25]).astype("float32")
        x = paddle.to_tensor(input)
        out = self._executed_api(x, scale=2.0, bias=3.0)
        np.testing.assert_array_equal(out.numpy(), input * 2.0 + 3.0)
        paddle.enable_static()


class TestScaleInplaceApiDygraph(TestScaleApiDygraph):
    def _executed_api(self, x, scale=1.0, bias=0.0):
        return x.scale_(scale, bias)


class TestScaleOpZeroNumelVariable(unittest.TestCase):
    def test_check_zero_numel_xpu(self):
        if paddle.is_compiled_with_xpu():
            paddle.disable_static()
            paddle.set_device('xpu')
            data = paddle.ones([0, 1])
            out = paddle.scale(data, 2)
            self.assertEqual(out.shape, data.shape)
            paddle.enable_static()


support_types = get_xpu_op_support_types('scale')
for stype in support_types:
    create_test_class(globals(), XPUTestScaleOp, stype)

if __name__ == "__main__":
    unittest.main()
