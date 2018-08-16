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
import numpy as np
from functools import partial
import paddle.fluid.core as core
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


def create_test_class(test_case, callback, attrs):
    class TestFusedElementwiseActivationOp_base(OpTest):
        def setUp(self):
            self.op_type = "fused_elemwise_activation"
            self.dtype = np.float32
            self.axis = -1

            self.init_input()
            self.init_output()
            self.init_attr()

            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(self.x),
                'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
            }
            if self.attrs["keep_intermediate_value"]:
                self.outputs = {
                    'Out': [('out', self.out),
                            ('intermediate_out', self.intermediate_out)]
                }
            else:
                self.outputs = {'Out': [('out', self.out)]}

        def init_input(self):
            self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
            self.axis = -1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y)

        def init_attr(self):
            self.attrs = {'axis': self.axis, }
            for key in attrs.keys():
                self.attrs[key] = attrs[key]

        def test_check_output(self):
            self.check_output()

        def test_check_grad_normal(self):
            self.check_grad(['X', 'Y'], ['out'], max_relative_error=0.005)
            # if self.attrs["keep_intermediate_value"]:
            #     self.check_grad(['X', 'Y'], ['out','intermediate_out'], max_relative_error=0.005)
            # else:

        def test_check_grad_ingore_x(self):
            self.check_grad(
                ['Y'], 'out', max_relative_error=0.005, no_grad_set=set("X"))

        def test_check_grad_ingore_y(self):
            self.check_grad(
                ['X'], 'out', max_relative_error=0.005, no_grad_set=set('Y'))

    class TestFusedElementwiseActivationOp_scalar(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)

    class TestFusedElementwiseActivationOp_scalar2(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(1, 1).astype(self.dtype)

    class TestFusedElementwiseActivationOp_Vector(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.random((32, )).astype(self.dtype)
            self.y = np.random.random((32, )).astype(self.dtype)

    class TestFusedElementwiseActivationOp_broadcast_0(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(2).astype(self.dtype)
            self.axis = 0

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(2, 1, 1))

    class TestFusedElementwiseActivationOp_broadcast_1(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(3).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(1, 3, 1))

    class TestFusedElementwiseActivationOp_broadcast_2(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(4).astype(self.dtype)

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(1, 1, 4))

    class TestFusedElementwiseActivationOp_broadcast_3(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
            self.y = np.random.rand(3, 4).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(1, 3, 4, 1))

    class TestFusedElementwiseActivationOp_broadcast_4(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4, 5).astype(self.dtype)
            self.y = np.random.rand(2, 1).astype(self.dtype)
            self.axis = 0

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(2, 1, 1, 1))

    class TestFusedElementwiseActivationOp_rowwise_add_0(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 3, 4).astype(self.dtype)
            self.y = np.random.rand(3, 4).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(1, 3, 4))

    class TestFusedElementwiseActivationOp_rowwise_add_1(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(2, 1).astype(self.dtype)
            self.y = np.random.rand(1).astype(self.dtype)
            self.axis = 1

        def init_output(self):
            self.x, self.y, self.intermediate_out, self.out = \
                callback(self.x, self.y, self.x, self.y.reshape(1, 1))

    class TestFusedElementwiseActivationOp_channelwise_add(
            TestFusedElementwiseActivationOp_base):
        def init_input(self):
            self.x = np.random.rand(3, 20, 20).astype(self.dtype)
            self.y = np.random.rand(3, 1, 1).astype(self.dtype)

    TestFusedElementwiseActivationOp_base.__name__ = test_case + "_base"
    TestFusedElementwiseActivationOp_scalar.__name__ = test_case + "_scalar"
    TestFusedElementwiseActivationOp_scalar2.__name__ = test_case + "_scalar2"
    TestFusedElementwiseActivationOp_Vector.__name__ = test_case + "_Vector"
    TestFusedElementwiseActivationOp_broadcast_0.__name__ = test_case + "_broadcast_0"
    TestFusedElementwiseActivationOp_broadcast_1.__name__ = test_case + "_broadcast_1"
    TestFusedElementwiseActivationOp_broadcast_2.__name__ = test_case + "_broadcast_2"
    TestFusedElementwiseActivationOp_broadcast_3.__name__ = test_case + "_broadcast_3"
    TestFusedElementwiseActivationOp_broadcast_4.__name__ = test_case + "_broadcast_4"
    TestFusedElementwiseActivationOp_rowwise_add_0.__name__ = test_case + "_rowwise_add_0"
    TestFusedElementwiseActivationOp_rowwise_add_1.__name__ = test_case + "_rowwise_add_1"
    TestFusedElementwiseActivationOp_channelwise_add.__name__ = test_case + "_channelwise_add"

    globals()[test_case + "_base"] = TestFusedElementwiseActivationOp_base
    globals()[test_case + "_scalar"] = TestFusedElementwiseActivationOp_scalar
    globals()[test_case + "_scalar2"] = TestFusedElementwiseActivationOp_scalar2
    globals()[test_case + "_Vector"] = TestFusedElementwiseActivationOp_Vector
    globals()[test_case +
              "_broadcast_0"] = TestFusedElementwiseActivationOp_broadcast_0
    globals()[test_case +
              "_broadcast_1"] = TestFusedElementwiseActivationOp_broadcast_1
    globals()[test_case +
              "_broadcast_2"] = TestFusedElementwiseActivationOp_broadcast_2
    globals()[test_case +
              "_broadcast_3"] = TestFusedElementwiseActivationOp_broadcast_3
    globals()[test_case +
              "_broadcast_4"] = TestFusedElementwiseActivationOp_broadcast_4
    globals()[test_case +
              "_rowwise_add_0"] = TestFusedElementwiseActivationOp_rowwise_add_0
    globals()[test_case +
              "_rowwise_add_1"] = TestFusedElementwiseActivationOp_rowwise_add_1
    globals(
    )[test_case +
      "_channelwise_add"] = TestFusedElementwiseActivationOp_channelwise_add


def scale_add_func(x, y, x_bcast, y_bcast, scale):
    return x, y, (x_bcast + y_bcast), (x_bcast + y_bcast) * scale


def add_scale_func(x, y, x_bcast, y_bcast, scale):
    return x, y, y * scale, x_bcast + y_bcast * scale


def add_relu_func(x, y, x_bcast, y_bcast):
    # Copy from test_activation_op.py
    # Because we set delta = 0.005 in calculating numeric gradient,
    # if x is too small, such as 0.002, x_neg will be -0.003
    # x_pos will be 0.007, so the numeric gradient is inaccurate.
    # we should avoid this
    y[np.abs(y) < 0.005] = 0.02
    y_bcast[np.abs(y_bcast) < 0.005] = 0.02
    return x, y, np.maximum(y, 0), x_bcast + np.maximum(y_bcast, 0)


def relu_add_func(x, y, x_bcast, y_bcast):
    intermediate_out = x_bcast + y_bcast
    out = np.maximum(intermediate_out, 0)
    out[np.abs(out) < 0.005] = 0.02
    return x, y, intermediate_out, out


scale = 0.1
scale_add_func = partial(scale_add_func, scale=scale)
add_scale_func = partial(add_scale_func, scale=scale)

for keep_intermediate_value in {False}:  # ,True
    kiv = keep_intermediate_value
    create_test_class('scale_add' + ("_keep_intermediate_value" if kiv else ""),
                      scale_add_func, {
                          'scale': scale,
                          'functor_list': ["scale", "elementwise_add"],
                          'keep_intermediate_value': kiv
                      })
    create_test_class('add_scale' + ("_keep_intermediate_value" if kiv else ""),
                      add_scale_func, {
                          'scale': scale,
                          'functor_list': ["elementwise_add", "scale"],
                          'keep_intermediate_value': keep_intermediate_value
                      })
    create_test_class('add_relu' + ("_keep_intermediate_value" if kiv else ""),
                      add_relu_func, {
                          'functor_list': ["elementwise_add", "relu"],
                          'keep_intermediate_value': keep_intermediate_value
                      })
    create_test_class('relu_add' + ("_keep_intermediate_value" if kiv else ""),
                      relu_add_func, {
                          'functor_list': ["relu", "elementwise_add"],
                          'keep_intermediate_value': keep_intermediate_value
                      })

if __name__ == '__main__':
    unittest.main()
