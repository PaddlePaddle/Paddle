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

import paddle
from paddle import base
from paddle.base import Program, core

paddle.enable_static()


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
        self.place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        self.x_np = np.ones([1, 2, 3, 4]).astype('float32')

    def test_mode_error(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'any')
            except Exception as e:
                assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error1(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = prelu_t(x, 'channel', data_format='N')
            except Exception as e:
                assert e.args[0].find('InvalidArgument') != -1

    def test_data_format_error2(self):
        main_program = Program()
        with base.program_guard(main_program, Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4, 5])
            try:
                y = paddle.static.nn.prelu(x, 'channel', data_format='N')
            except ValueError as e:
                pass


if __name__ == "__main__":
    unittest.main()
