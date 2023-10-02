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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle import _legacy_C_ops, base
from paddle.base import core
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import in_dygraph_mode
from paddle.base.layer_helper import LayerHelper
from paddle.static import default_main_program


def dropout_nd(
    x, p=0.5, axis=None, training=True, mode="upscale_in_train", name=None
):
    drop_axes = [axis] if isinstance(axis, int) else list(axis)
    seed = None
    mode = (
        'downgrade_in_infer' if mode == 'downscale_in_infer' else mode
    )  # semantic transfer
    if in_dygraph_mode():
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
        x, 'x', ['float16', 'float32', 'float64', 'uint16'], 'dropout'
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
        # NODE(yjjiang11): This op will be deprecated.
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        # NODE(yjjiang11): This op will be deprecated.
        self.check_grad(['X'], 'Out', check_dygraph=False)


class TestDropoutNdFP16Op(OpTest):
    def setUp(self):
        self.op_type = "dropout_nd"
        self.dtype = np.float16
        self.inputs = {'X': np.random.random((2, 16, 8)).astype("float16")}
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'axis': [1],
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((1, 16, 1)).astype('uint8'),
        }

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', check_dygraph=False)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not complied with CUDA and not support the bfloat16",
)
class TestDropoutNdBF16Op(OpTest):
    def setUp(self):
        self.op_type = "dropout_nd"
        self.dtype = np.uint16
        self.np_dtype = "float32"
        self.inputs = {
            'X': convert_float_to_uint16(
                np.random.random((2, 16, 8)).astype(self.np_dtype)
            )
        }
        self.attrs = {
            'dropout_prob': 0.0,
            'fix_seed': True,
            'is_test': False,
            'axis': [1],
        }
        self.outputs = {
            'Out': self.inputs['X'],
            'Mask': np.ones((1, 16, 1)).astype('uint8'),
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), check_dygraph=False)

    def test_check_grad_normal(self):
        self.check_grad_with_place(
            core.CUDAPlace(0), ['X'], 'Out', check_dygraph=False
        )


class TestDropoutNdAPI(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        np.random.seed(123)
        self.places = [base.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def test_dygraph(self):
        paddle.disable_static()
        for place in self.places:
            with base.dygraph.guard(place):
                in_np = np.random.random([4, 32, 16]).astype("float32")
                input = paddle.to_tensor(in_np)
                dropout_1 = paddle.incubate.nn.FusedDropout(p=0.0, axis=[0, 1])
                dropout_2 = paddle.incubate.nn.FusedDropout(p=0.5, axis=[0, 1])
                print(dropout_1)
                print(dropout_2)
                res1 = dropout_1(input)
                res2 = dropout_2(input)
            np.testing.assert_allclose(res1.numpy(), in_np, rtol=1e-05)
        paddle.enable_static()

    def test_error(self):
        def _run_illegal_type_p():
            dropout = paddle.incubate.nn.FusedDropout(p="test")

        self.assertRaises(TypeError, _run_illegal_type_p)

        def _run_illegal_value_p():
            dropout = paddle.incubate.nn.FusedDropout(p=2)

        self.assertRaises(ValueError, _run_illegal_value_p)

        def _run_illegal_mode():
            dropout = paddle.incubate.nn.FusedDropout(p=0.5, mode="test")

        self.assertRaises(ValueError, _run_illegal_mode)

        def _run_illegal_type_axis():
            dropout = paddle.incubate.nn.FusedDropout(p=0.5, axis="test")

        self.assertRaises(TypeError, _run_illegal_type_axis)


if __name__ == '__main__':
    unittest.main()
