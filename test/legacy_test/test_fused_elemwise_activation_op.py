#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from functools import partial

import numpy as np
from op_test import OpTest

#   TestFusedElementwiseActivationOp
#   TestFusedElementwiseActivationOp_scalar
#   TestFusedElementwiseActivationOp_scalar2
#   TestFusedElementwiseActivationOp_Vector
#   TestFusedElementwiseActivationOp_broadcast_0
#   TestFusedElementwiseActivationOp_broadcast_1
#   TestFusedElementwiseActivationOp_broadcast_2
#   TestFusedElementwiseActivationOp_broadcast_3
#   TestFusedElementwiseActivationOp_broadcast_4
#   TestFusedElementwiseActivationOp_rowwise_add_0
#   TestFusedElementwiseActivationOp_rowwise_add_1
#   TestFusedElementwiseActivationOp_channelwise_add
import paddle
from paddle.base import core


def api_wrapper(
    x, y, functor_list=[], axis=-1, scale=0.0, save_intermediate_out=False
):
    return paddle._legacy_C_ops.fused_elemwise_activation(
        x,
        y,
        "axis",
        axis,
        "scale",
        scale,
        "save_intermediate_out",
        save_intermediate_out,
        "functor_list",
        functor_list,
    )


def create_test_class(
    test_case, callback, attrs, dtype=np.float32, grad_check=True
):
    class TestFusedElementwiseActivationOp_base(OpTest):
        def setUp(self):
            self.op_type = "fused_elemwise_activation"
            self.python_api = api_wrapper
            self.python_out_sig = ['Out']
            self.dtype = dtype
            self.axis = -1

            self.init_input()
            self.init_output()
            self.init_attr()

            self.out = self.out.astype(self.dtype)
            self.intermediate_out = self.intermediate_out.astype(self.dtype)

            self.inputs = {
                'X': OpTest.np_dtype_to_base_dtype(self.x),
                'Y': OpTest.np_dtype_to_base_dtype(self.y),
            }
            if self.attrs["save_intermediate_out"]:
                self.outputs = {
                    'Out': self.out,
                    "IntermediateOut": self.intermediate_out,
                }
            else:
                self.outputs = {'Out': self.out}

        def init_input(self):
            self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.axis = -1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y
            )

        def init_attr(self):
            self.attrs = {
                'axis': self.axis,
            }
            for key in attrs.keys():
                self.attrs[key] = attrs[key]

        def test_check_output(self):
            if self.dtype == np.float16 and core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place, atol=1e-3)
            else:
                self.check_output()

        # FIXME(zcd): the intermediate_out_grad is not checked.
        def test_check_grad_normal(self):
            if not grad_check:
                return
            if self.attrs["save_intermediate_out"]:
                self.check_grad(['X', 'Y'], ['Out'], check_dygraph=False)
            else:
                self.check_grad(['X', 'Y'], ['Out'], check_dygraph=False)

        def test_check_grad_ignore_x(self):
            if not grad_check:
                return
            if self.attrs["save_intermediate_out"]:
                self.check_grad(
                    ['Y'],
                    ['Out'],
                    max_relative_error=0.005,
                    no_grad_set=set("X"),
                    check_dygraph=False,
                )
            else:
                self.check_grad(
                    ['Y'],
                    ['Out'],
                    max_relative_error=0.005,
                    no_grad_set=set("X"),
                    check_dygraph=False,
                )

        def test_check_grad_ignore_y(self):
            if not grad_check:
                return
            if self.attrs["save_intermediate_out"]:
                self.check_grad(
                    ['X'],
                    ['Out'],
                    max_relative_error=0.005,
                    no_grad_set=set("Y"),
                    check_dygraph=False,
                )
            else:
                self.check_grad(
                    ['X'],
                    ['Out'],
                    max_relative_error=0.005,
                    no_grad_set=set("Y"),
                    check_dygraph=False,
                )

    class TestFusedElementwiseActivationOp_scalar(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)

    class TestFusedElementwiseActivationOp_scalar2(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1, 1).astype(self.dtype)

    class TestFusedElementwiseActivationOp_Vector(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.random((32,)).astype(self.dtype)
            self.y = np.random.random((32,)).astype(self.dtype)

    class TestFusedElementwiseActivationOp_broadcast_0(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(2).astype(self.dtype)
            self.axis = 0

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(2, 1, 1)
            )

    class TestFusedElementwiseActivationOp_broadcast_1(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(3).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(1, 3, 1)
            )

    class TestFusedElementwiseActivationOp_broadcast_2(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(4).astype(self.dtype)

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(1, 1, 4)
            )

    class TestFusedElementwiseActivationOp_broadcast_3(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
            self.y = np.random.rand(3, 4).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(1, 3, 4, 1)
            )

    class TestFusedElementwiseActivationOp_broadcast_4(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
            self.y = np.random.rand(2, 1).astype(self.dtype)
            self.axis = 0

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(2, 1, 1, 1)
            )

    class TestFusedElementwiseActivationOp_rowwise_add_0(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(3, 4).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(1, 3, 4)
            )

    class TestFusedElementwiseActivationOp_rowwise_add_1(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(2, 1).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = callback(
                self.x, self.y, self.x, self.y.reshape(1, 1)
            )

    class TestFusedElementwiseActivationOp_channelwise_add(
        TestFusedElementwiseActivationOp_base
    ):
        def init_input(self):
            self.x = np.random.rand(3, 20, 20).astype(self.dtype)
            self.y = np.random.rand(3, 1, 1).astype(self.dtype)

    TestFusedElementwiseActivationOp_base.__name__ = test_case + "_base"
    TestFusedElementwiseActivationOp_scalar.__name__ = test_case + "_scalar"
    TestFusedElementwiseActivationOp_scalar2.__name__ = test_case + "_scalar2"
    TestFusedElementwiseActivationOp_Vector.__name__ = test_case + "_Vector"
    TestFusedElementwiseActivationOp_broadcast_0.__name__ = (
        test_case + "_broadcast_0"
    )
    TestFusedElementwiseActivationOp_broadcast_1.__name__ = (
        test_case + "_broadcast_1"
    )
    TestFusedElementwiseActivationOp_broadcast_2.__name__ = (
        test_case + "_broadcast_2"
    )
    TestFusedElementwiseActivationOp_broadcast_3.__name__ = (
        test_case + "_broadcast_3"
    )
    TestFusedElementwiseActivationOp_broadcast_4.__name__ = (
        test_case + "_broadcast_4"
    )
    TestFusedElementwiseActivationOp_rowwise_add_0.__name__ = (
        test_case + "_rowwise_add_0"
    )
    TestFusedElementwiseActivationOp_rowwise_add_1.__name__ = (
        test_case + "_rowwise_add_1"
    )
    TestFusedElementwiseActivationOp_channelwise_add.__name__ = (
        test_case + "_channelwise_add"
    )

    globals()[test_case + "_base"] = TestFusedElementwiseActivationOp_base
    globals()[test_case + "_scalar"] = TestFusedElementwiseActivationOp_scalar
    globals()[test_case + "_scalar2"] = TestFusedElementwiseActivationOp_scalar2
    globals()[test_case + "_Vector"] = TestFusedElementwiseActivationOp_Vector
    globals()[
        test_case + "_broadcast_0"
    ] = TestFusedElementwiseActivationOp_broadcast_0
    globals()[
        test_case + "_broadcast_1"
    ] = TestFusedElementwiseActivationOp_broadcast_1
    globals()[
        test_case + "_broadcast_2"
    ] = TestFusedElementwiseActivationOp_broadcast_2
    globals()[
        test_case + "_broadcast_3"
    ] = TestFusedElementwiseActivationOp_broadcast_3
    globals()[
        test_case + "_broadcast_4"
    ] = TestFusedElementwiseActivationOp_broadcast_4
    globals()[
        test_case + "_rowwise_add_0"
    ] = TestFusedElementwiseActivationOp_rowwise_add_0
    globals()[
        test_case + "_rowwise_add_1"
    ] = TestFusedElementwiseActivationOp_rowwise_add_1
    globals()[
        test_case + "_channelwise_add"
    ] = TestFusedElementwiseActivationOp_channelwise_add


def scale_add_func(x, y, x_bcast, y_bcast, scale, mode=0):
    if mode == 0:
        return x, y, (x_bcast + y_bcast), (x_bcast + y_bcast) * scale
    else:
        return y, x, (x_bcast + y_bcast), (x_bcast + y_bcast) * scale


def add_scale_func(x, y, x_bcast, y_bcast, scale, mode=0):
    if mode == 0:
        return x, y, y * scale, x_bcast + y_bcast * scale
    else:
        return y, x, x * scale, y_bcast + x_bcast * scale


def add_relu_func(x, y, x_bcast, y_bcast, mode=0):
    # Copy from test_activation_op.py
    # Because we set delta = 0.005 in calculating numeric gradient,
    # if x is too small, such as 0.002, x_neg will be -0.003
    # x_pos will be 0.007, so the numeric gradient is inaccurate.
    # we should avoid this
    if mode == 0:
        y[np.abs(y) < 0.005] = 0.02
        y_bcast[np.abs(y_bcast) < 0.005] = 0.02
        return x, y, np.maximum(y, 0), x_bcast + np.maximum(y_bcast, 0)
    else:
        x[np.abs(x) < 0.005] = 0.02
        x_bcast[np.abs(x_bcast) < 0.005] = 0.02
        return y, x, np.maximum(x, 0), y_bcast + np.maximum(x_bcast, 0)


def relu_add_func(x, y, x_bcast, y_bcast, mode=0):
    intermediate_out = x_bcast + y_bcast
    out = np.maximum(intermediate_out, 0)
    out[np.abs(out) < 0.005] = 0.02
    if mode == 0:
        return x, y, intermediate_out, out
    else:
        return y, x, intermediate_out, out


def mul_scale_func(x, y, x_bcast, y_bcast, scale, mode=0):
    if mode == 0:
        return x, y, y * scale, x_bcast * (y_bcast * scale)
    else:
        return y, x, x * scale, y_bcast * (x_bcast * scale)


def gelu_add_func(x, y, x_bcast, y_bcast, mode=0):
    im = x_bcast + y_bcast
    out = im * 0.5 * (1.0 + np.tanh(0.79788456 * im * (1 + 0.044715 * im * im)))
    if mode == 0:
        return x, y, im, out
    else:
        return y, x, im, out


scale = 0.1
scale_add_func = partial(scale_add_func, scale=scale)
add_scale_func = partial(add_scale_func, scale=scale)
mul_scale_func = partial(mul_scale_func, scale=scale)

for mode in {0, 1}:
    scale_add_func = partial(scale_add_func, mode=mode)
    add_scale_func = partial(add_scale_func, mode=mode)
    mul_scale_func = partial(mul_scale_func, mode=mode)
    relu_add_func = partial(relu_add_func, mode=mode)
    add_relu_func = partial(add_relu_func, mode=mode)
    gelu_add_func = partial(gelu_add_func, mode=mode)

    for save_intermediate_out in {True, False}:
        suffix = ("_save_intermediate_out" if save_intermediate_out else "") + (
            "_mode_" + str(mode)
        )
        create_test_class(
            'scale_add' + suffix,
            scale_add_func,
            {
                'scale': scale,
                'functor_list': ["scale", "elementwise_add"],
                'save_intermediate_out': save_intermediate_out,
            },
        )
        create_test_class(
            'add_scale' + suffix,
            add_scale_func,
            {
                'scale': scale,
                'functor_list': ["elementwise_add", "scale"],
                'save_intermediate_out': save_intermediate_out,
            },
        )
        create_test_class(
            'add_relu' + suffix,
            add_relu_func,
            {
                'functor_list': ["elementwise_add", "relu"],
                'save_intermediate_out': save_intermediate_out,
            },
        )
        create_test_class(
            'relu_add' + suffix,
            relu_add_func,
            {
                'functor_list': ["relu", "elementwise_add"],
                'save_intermediate_out': save_intermediate_out,
            },
        )
        create_test_class(
            'mul_scale' + suffix,
            mul_scale_func,
            {
                'scale': scale,
                'functor_list': ["elementwise_mul", "scale"],
                'save_intermediate_out': save_intermediate_out,
            },
        )
        create_test_class(
            'gelu_add' + suffix,
            gelu_add_func,
            {
                'functor_list': ["gelu", "elementwise_add"],
                'save_intermediate_out': save_intermediate_out,
            },
        )

        if core.is_compiled_with_cuda():
            create_test_class(
                'scale_add_fp16' + suffix,
                scale_add_func,
                {
                    'scale': scale,
                    'functor_list': ["scale", "elementwise_add"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )
            create_test_class(
                'add_scale_fp16' + suffix,
                add_scale_func,
                {
                    'scale': scale,
                    'functor_list': ["elementwise_add", "scale"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )

            create_test_class(
                'add_relu_fp16' + suffix,
                add_relu_func,
                {
                    'functor_list': ["elementwise_add", "relu"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )
            create_test_class(
                'relu_add_fp16' + suffix,
                relu_add_func,
                {
                    'functor_list': ["relu", "elementwise_add"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )
            create_test_class(
                'mul_scale_fp16' + suffix,
                mul_scale_func,
                {
                    'scale': scale,
                    'functor_list': ["elementwise_mul", "scale"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )
            create_test_class(
                'gelu_add_fp16' + suffix,
                gelu_add_func,
                {
                    'functor_list': ["gelu", "elementwise_add"],
                    'save_intermediate_out': save_intermediate_out,
                },
                dtype=np.float16,
                grad_check=False,
            )

if __name__ == '__main__':
    import paddle

    paddle.enable_static()
    unittest.main()
