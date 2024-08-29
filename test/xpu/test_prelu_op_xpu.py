#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
    create_test_class,
    get_xpu_op_support_types,
)
from op_test_xpu import XPUOpTest

import paddle
from paddle import base

paddle.enable_static()


class XPUTestPReluOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "prelu"
        self.use_dynamic_create_class = False

    class TestPReluOp(XPUOpTest):
        def setUp(self):
            self.set_xpu()
            self.op_type = "prelu"
            self.init_dtype()

            # override
            self.init_input_shape()
            self.init_attr()

            self.x = np.random.uniform(-10.0, 10.0, self.x_shape).astype(
                self.dtype
            )
            # Since zero point in prelu is not differentiable, avoid randomize zero.
            self.x[np.abs(self.x) < 0.005] = 0.02

            if self.attrs == {
                'mode': "all",
                "data_format": "NCHW",
            } or self.attrs == {'mode': "all", "data_format": "NHWC"}:
                self.alpha = np.random.uniform(-1, -0.5, (1))
            elif self.attrs == {'mode': "channel", "data_format": "NCHW"}:
                self.alpha = np.random.uniform(
                    -1, -0.5, [1, self.x_shape[1], 1, 1]
                )
            elif self.attrs == {'mode': "channel", "data_format": "NHWC"}:
                self.alpha = np.random.uniform(
                    -1, -0.5, [1, 1, 1, self.x_shape[-1]]
                )
            else:
                self.alpha = np.random.uniform(-1, -0.5, [1] + self.x_shape[1:])
            self.alpha = self.alpha.astype(self.dtype)

            self.inputs = {'X': self.x, 'Alpha': self.alpha}

            reshaped_alpha = self.inputs['Alpha']
            if self.attrs == {'mode': "channel", "data_format": "NCHW"}:
                reshaped_alpha = np.reshape(
                    self.inputs['Alpha'],
                    [1, self.x_shape[1]] + [1] * len(self.x_shape[2:]),
                )
            elif self.attrs == {'mode': "channel", "data_format": "NHWC"}:
                reshaped_alpha = np.reshape(
                    self.inputs['Alpha'],
                    [1] + [1] * len(self.x_shape[1:-1]) + [self.x_shape[-1]],
                )

            self.alpha = np.random.uniform(
                -10.0, 10.0, [1, self.x_shape[1], 1, 1]
            ).astype(self.dtype)

            out_np = np.maximum(self.inputs['X'], 0.0)
            out_np = out_np + np.minimum(self.inputs['X'], 0.0) * reshaped_alpha
            assert out_np is not self.inputs['X']
            self.outputs = {'Out': out_np}

        def init_input_shape(self):
            self.x_shape = [2, 3, 5, 6]

        def init_attr(self):
            self.attrs = {'mode': "channel", 'data_format': "NCHW"}

        def set_xpu(self):
            self.__class__.no_need_check_grad = False
            self.place = paddle.XPUPlace(0)

        def init_dtype(self):
            self.dtype = self.in_type

        def test_check_output(self):
            self.check_output_with_place(self.place)

        def test_check_grad(self):
            self.check_grad_with_place(
                self.place, ['X', 'Alpha'], 'Out', check_dygraph=False
            )

    class TestModeChannelNHWC(TestPReluOp):
        def init_input_shape(self):
            self.x_shape = [2, 3, 4, 5]

        def init_attr(self):
            self.attrs = {'mode': "channel", "data_format": "NHWC"}

    class TestModeAll(TestPReluOp):
        def init_input_shape(self):
            self.x_shape = [2, 3, 4, 5]

        def init_attr(self):
            self.attrs = {'mode': "all", "data_format": "NCHW"}

    class TestModeAllNHWC(TestPReluOp):
        def init_input_shape(self):
            self.x_shape = [2, 3, 4, 50]

        def init_attr(self):
            self.attrs = {'mode': "all", "data_format": "NHWC"}

    class TestModeElt(TestPReluOp):
        def init_input_shape(self):
            self.x_shape = [3, 2, 5, 10]

        def init_attr(self):
            self.attrs = {'mode': "element", "data_format": "NCHW"}

    class TestModeEltNHWC(TestPReluOp):
        def init_input_shape(self):
            self.x_shape = [3, 2, 5, 10]

        def init_attr(self):
            self.attrs = {'mode': "element", "data_format": "NHWC"}


def prelu_t(x, mode, param_attr=None, name=None, data_format='NCHW'):
    helper = base.layer_helper.LayerHelper('prelu', **locals())
    alpha_shape = [1, x.shape[1], 1, 1]
    dtype = helper.input_dtype(input_param_name='x')
    alpha = helper.create_parameter(
        attr=helper.param_attr,
        shape=alpha_shape,
        dtype='float32',
        is_bias=False,
        default_initializer=paddle.nn.initializer.Constant(0.25),
    )
    out = helper.create_variable_for_type_inference(dtype)
    helper.append_op(
        type="prelu",
        inputs={"X": x, 'Alpha': alpha},
        attrs={"mode": mode, 'data_format': data_format},
        outputs={"Out": out},
    )
    return out


# error message test if mode is not one of 'all', 'channel', 'element'
class TestModeError(unittest.TestCase):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.x_np = np.ones([1, 2, 3, 4]).astype('float32')

    def test_mode_error(self):
        with paddle.pir_utils.OldIrGuard():
            main_program = paddle.base.Program()
            with base.program_guard(main_program, paddle.base.Program()):
                x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
                try:
                    y = prelu_t(x, 'any')
                except Exception as e:
                    assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error1(self):
        with paddle.pir_utils.OldIrGuard():
            main_program = paddle.base.Program()
            with base.program_guard(main_program, paddle.base.Program()):
                x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
                try:
                    y = prelu_t(x, 'channel', data_format='N')
                except Exception as e:
                    assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error2(self):
        main_program = paddle.base.Program()
        with base.program_guard(main_program, paddle.base.Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = paddle.nn.PReLU(3, data_format='N')(x)
            except ValueError as e:
                pass


support_types = get_xpu_op_support_types("prelu")
for stype in support_types:
    create_test_class(globals(), XPUTestPReluOp, stype)

if __name__ == "__main__":
    unittest.main()
