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
from op_test_xpu import XPUOpTest

import paddle

paddle.enable_static()


def compute_index_add_ref(
    axis, x_shape, x_np, add_value_shape, add_value_np, index_size, index_np
):
    if axis < 0:
        axis = axis + len(x_shape)
    if axis != 0:
        outer_loop = np.prod(x_shape[:axis]).astype(int)
        x_reshape = [outer_loop, *x_shape[axis:]]
        x_np_reshape = np.reshape(x_np, tuple(x_reshape))

        add_value_reshape = [
            np.prod(add_value_shape[:axis]).astype(int),
            *add_value_shape[axis:],
        ]

        add_value_np_reshape = np.reshape(
            add_value_np, tuple(add_value_reshape)
        )
    else:
        x_np_reshape = x_np
        add_value_np_reshape = add_value_np
    out_np = x_np_reshape.copy()

    if axis != 0:
        for i in range(outer_loop):
            for j in range(index_size):
                out_np[i, index_np[j]] += add_value_np_reshape[i, j]
    else:
        for j in range(index_size):
            out_np[index_np[j]] += add_value_np_reshape[j]
    ref_out = np.reshape(out_np, x_shape)
    return ref_out


def raw_index_add(x, index, value, axis):
    return paddle.index_add(x, index, axis, value)


class XPUTestIndexAddOp(XPUOpTest):
    def setUp(self):
        self.python_api = raw_index_add
        self.op_type = "index_add"
        self.init_dtype_type()
        index_np = np.random.randint(
            low=0, high=self.x_shape[self.axis], size=self.index_size
        )
        x_np = np.random.random(self.x_shape).astype(self.x_type)
        add_value_np = np.random.random(self.add_value_shape).astype(
            self.x_type
        )

        self.inputs = {'X': x_np, 'Index': index_np, 'AddValue': add_value_np}
        self.attrs = {'axis': self.axis}
        out = compute_index_add_ref(
            self.axis,
            self.x_shape,
            x_np,
            self.add_value_shape,
            add_value_np,
            self.index_size,
            index_np,
        )
        self.outputs = {'Out': out}

    def init_dtype_type(self):
        self.axis = 0
        self.x_type = np.float64
        self.index_type = np.int64
        self.x_shape = (101, 3)
        self.index_size = 3
        self.add_value_shape = (3, 3)

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            self.check_output_with_place(
                paddle.XPUPlace(0),
                atol=1e-4,
                check_dygraph=False,
            )

    def test_check_grad_normal(self):
        if paddle.is_compiled_with_xpu():
            self.check_grad_with_place(
                paddle.XPUPlace(0),
                ['X', 'AddValue'],
                'Out',
                max_relative_error=1e-4,
                check_dygraph=False,
            )


if __name__ == '__main__':
    unittest.main()
