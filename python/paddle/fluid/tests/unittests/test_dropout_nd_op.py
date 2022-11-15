#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.framework import _non_static_mode
from paddle import _legacy_C_ops
from paddle.static import default_main_program
from paddle.fluid.data_feeder import check_variable_and_dtype


def dropout_nd(
    x, p=0.5, axis=None, training=True, mode="upscale_in_train", name=None
):
    drop_axes = [axis] if isinstance(axis, int) else list(axis)
    seed = None
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer
    if _non_static_mode():
        if default_main_program().random_seed != 0:
            seed = default_main_program().random_seed

        out, mask = _legacy_C_ops.dropout_nd(
            x,
            'dropout_prob',
            p,
            'is_test',
            not training,
            'fix_seed',
            seed is not None,
            'seed',
            seed if seed is not None else 0,
            'dropout_implementation',
            mode,
            'axis',
            drop_axes,
        )
        return out

    helper = LayerHelper('dropout_nd', **locals())
    check_variable_and_dtype(
        x, 'x', ['float16', 'float32', 'float64'], 'dropout'
    )

    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    mask = helper.create_variable_for_type_inference(
        dtype=core.VarDesc.VarType.UINT8, stop_gradient=True
    )

    def get_attrs(prog, dropout_prob, is_test, seed):
        if (seed is None or seed == 0) and prog.random_seed != 0:
            seed = prog.random_seed
        attrs = {
            'dropout_prob': dropout_prob,
            'is_test': is_test,
            'fix_seed': seed is not None,
            'seed': seed if seed is not None else 0,
            'dropout_implementation': mode,
            'axis': drop_axes,
        }
        return attrs

    attrs = get_attrs(helper.main_program, p, not training, seed)

    helper.append_op(
        type='dropout_nd',
        inputs={'X': [x]},
        outputs={'Out': [out], 'Mask': [mask]},
        attrs=attrs,
    )
    return out


paddle.enable_static()


class TestDropoutNdOp(OpTest):
    def setUp(self):
        self.op_type = "dropout_nd"
        self.inputs = {'X': np.random.random((4, 32, 16)).astype("float64")}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'axis': [1],
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((1, 32, 1)).astype('uint8'),
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestDropoutNdAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            with fluid.dygraph.guard(place):
                in_np = np.random.random([4, 32, 16]).astype("float32")
                input = paddle.to_tensor(in_np)
                res1 = dropout_nd(x=input, p=0.0, axis=[0, 1])
                res2 = dropout_nd(x=input, p=0.5, axis=[0, 1])
            np.testing.assert_allclose(res1.numpy(), in_np, rtol=1e-05)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
