#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import _C_ops
from paddle.base.data_feeder import (
    check_type,
    check_variable_and_dtype,
    convert_dtype,
)
from paddle.base.framework import Variable
from paddle.framework import (
    LayerHelper,
    in_dynamic_or_pir_mode,
)
from paddle.static import InputSpec


def expand_as_net(x, y, is_origin=True):
    if in_dynamic_or_pir_mode():
        if is_origin:
            return _C_ops.expand_as(x, None, y.shape)
        else:
            return _C_ops.expand_as(x, y, [20, 3])
    else:
        check_variable_and_dtype(
            x,
            'x',
            [
                'bool',
                'float32',
                'float64',
                'int32',
                'int64',
                'float16',
                'uint16',
            ],
            'expand_as',
        )
        check_type(y, 'y', Variable, 'expand_as')

        if convert_dtype(x.dtype) == 'bool' and not x.stop_gradient:
            raise ValueError(
                "When the data type of input 'x' for expand_as is bool, "
                "you must set its stop_gradient to be False by "
                "some_var.stop_gradient = True, supporting "
                "some_var as the input 'x'."
            )
        inputs = {"X": [x], "Y": [y]}

        helper = LayerHelper('expand_as', **locals())
        dtype = helper.input_dtype(input_param_name='x')
        out = helper.create_variable_for_type_inference(dtype)
        if is_origin:
            target_shape = y.shape
        else:
            target_shape = [20, 3]
        helper.append_op(
            type='expand_as_v2',
            inputs=inputs,
            attrs={'target_shape': target_shape},
            outputs={'Out': out},
        )
        return out


def get_static_output(x, y, is_origin=True):
    build_strategy = paddle.static.BuildStrategy()
    input_spec = [
        InputSpec(shape=x.shape, dtype=x.dtype),
        InputSpec(shape=y.shape, dtype=y.dtype),
    ]
    fn = paddle.jit.to_static(
        expand_as_net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )
    return fn(x, y, is_origin)


class TestExpandAsKernelInput(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.x_dtype = "float32"
        self.y_dtype = "float32"
        self.x_shape = [3]
        self.y_shape = [2, 3]
        self.x = np.random.random(self.x_shape).astype(self.x_dtype)
        self.y = np.random.random(self.y_shape).astype(self.y_dtype)
        self.net = expand_as_net

    def test_dygraph_mode(self):
        paddle.disable_static()
        x_data = paddle.to_tensor(self.x)
        y_data = paddle.to_tensor(self.y)
        out_origin = self.net(x_data, y_data, is_origin=True)
        out_new = self.net(x_data, y_data, is_origin=False)
        assert out_origin.equal_all(out_new).item()

    def test_static_mode(self):
        x_data = paddle.to_tensor(self.x)
        y_data = paddle.to_tensor(self.y)
        out_origin = get_static_output(x_data, y_data, is_origin=True)
        out_new = get_static_output(x_data, y_data, is_origin=False)
        assert out_origin.equal_all(out_new).item()


if __name__ == "__main__":
    unittest.main()
