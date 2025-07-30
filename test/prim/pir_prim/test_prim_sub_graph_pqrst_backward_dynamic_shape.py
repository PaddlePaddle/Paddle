# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from test_prim_sub_graph_backward_dynamic_shape import (
    TestPrimBaseOneGradTwoInputs,
    TestPrimBaseWithGrad,
    TestPrimThreeWithGrad,
    TestPrimTwoWithGrad,
    apply_to_static,
)

import paddle
from paddle.framework import core
from paddle.static import InputSpec


def pad_net(x):
    return paddle.nn.functional.pad(x, [0, 1, 2])


def pow_net(x):
    return paddle.pow(x, 3.2)


def prod_net1(x):
    return paddle.prod(x)


def prod_net2(x):
    return paddle.prod(x, 0)


def prod_net3(x):
    return paddle.prod(x, keepdim=False)


def prod_net4(x):
    return paddle.prod(x, 0, keepdim=False)


def reduce_as_net(x, y):
    return paddle.reduce_as(x, y)


def relu_net(x):
    return paddle.nn.functional.relu(x)


def relu6_net(x):
    return paddle.nn.functional.relu6(x)


def reshape_net(x):
    return paddle.reshape(x, [30, 200 * 40])


def roll_net(x):
    return paddle.roll(x, shifts=[101, -1], axis=[0, -2])


def scale_net(x):
    return paddle.scale(x, scale=-2.3)


def scatter_net(x, y, z):
    return paddle.scatter(x, y, z)


def scatter_nd_add_net(x, y, z):
    return paddle.scatter_nd_add(x, y, z)


def sigmoid_net(x):
    return paddle.nn.functional.sigmoid(x)


def softmax_net(x):
    return paddle.nn.functional.softmax(x, axis=-1)


def softsign_net(x):
    return paddle.nn.functional.softsign(x)


def split_net1(x):
    res = paddle.split(x, num_or_sections=10, axis=-1)
    tmp_res = res[0]
    for i in range(1, len(res)):
        tmp_res = tmp_res + res[i] * i
    return tmp_res / len(res)


def split_net2(x):
    res = paddle.split(x, num_or_sections=10, axis=1)
    tmp_res = res[0]
    for i in range(1, len(res)):
        tmp_res = tmp_res + res[i] * i
    return tmp_res / len(res)


def square_net(x):
    return paddle.square(x)


def squeeze_net(x):
    return paddle.squeeze(x, axis=[0, -2])


def stack_net1(x):
    y = x + 1
    return paddle.stack([x, y], axis=-1)


def stack_net2(x):
    y = x + 1
    return paddle.stack([x, y], axis=1)


def stack_net3(x):
    return paddle.stack(x, axis=0)


def subtract_net(x, y):
    return x - y


def sum_net1(x):
    return paddle.sum(x, axis=1, keepdim=False)


def sum_net2(x):
    return paddle.sum(x)


def sum_net3(x):
    return paddle.sum(x, keepdim=True)


def sum_net4(x):
    return paddle.sum(x, axis=-1, keepdim=False)


def sum_net5(x):
    return paddle.sum(x, axis=[0, 2], keepdim=False)


def swiglu_net1(x, y):
    return paddle.incubate.nn.functional.swiglu(x, y)


def swiglu_net2(x):
    return paddle.incubate.nn.functional.swiglu(x)


def swish_net(x):
    return paddle.nn.functional.swish(x)


def take_along_axis_net1(x, y):
    return paddle.take_along_axis(x, y, 0, True)


def take_along_axis_net2(x, y):
    return paddle.take_along_axis(x, y, 1, True)


def take_along_axis_net3(x, y):
    return paddle.take_along_axis(x, y, 1, False)


def tanh_net(x):
    return paddle.tanh(x)


def tile_net1(x):
    return paddle.tile(x, [2, 1, 3, 5, 4])


def tile_net2(x):
    return paddle.tile(x, [2, 2])


def tile_net3(x):
    return paddle.tile(x, [2, 3, 4])


def tile_net4(x):
    return paddle.tile(x, [5])


def topk_net(x):
    return paddle.topk(x, k=3, axis=-1)[0]


def transpose_net(x):
    return paddle.transpose(x, perm=[0, 3, 1, 2])


def trunc_net(x):
    return paddle.trunc(x)


def p_norm_net1(x):
    return paddle.linalg.norm(x, p=2, axis=-1)


def p_norm_net2(x):
    return paddle.linalg.norm(x, p=2, axis=0)


def p_norm_net3(x):
    return paddle.linalg.norm(x, p=2, axis=1)


def p_norm_net4(x):
    return paddle.linalg.norm(x, p=1, axis=1)


def p_norm_net5(x):
    return paddle.linalg.norm(x, p=3, axis=None)


def p_norm_net6(x):
    return paddle.linalg.norm(x, p=3.5, axis=-1)


def p_norm_net7(x):
    return paddle.linalg.norm(x, p=-1, axis=None)


def p_norm_net8(x):
    return paddle.linalg.norm(x, p=-float("inf"), axis=None)


def p_norm_net9(x):
    return paddle.linalg.norm(x, p=float("inf"), axis=None)


class TestPrimPadWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.pad_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30, 40]
        self.init_x_shape = [None, None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = pad_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPowWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.pow_grad"
        self.dtype = "float32"
        self.x_shape = [100, 20, 30]
        self.init_x_shape = [None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = pow_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimProdWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.prod_grad"
        self.dtype = "float32"
        self.x_shape = [100]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = prod_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimProdWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.prod_grad"
        self.dtype = "float32"
        self.x_shape = [100, 20, 30]
        self.init_x_shape = [None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = prod_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimProdWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.prod_grad"
        self.dtype = "float32"
        self.x_shape = [100, 20, 30]
        self.init_x_shape = [None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = prod_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimProdWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.prod_grad"
        self.dtype = "float32"
        self.x_shape = [100, 20, 30]
        self.init_x_shape = [None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = prod_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimProdWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.prod_grad"
        self.dtype = "float32"
        self.x_shape = [100, 20, 30]
        self.init_x_shape = [None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = prod_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimReduceAsWithGrad2(TestPrimBaseOneGradTwoInputs):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.reduce_as_grad"
        self.dtype = "float32"
        self.y_shape = [30, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = reduce_as_net
        self.enable_cinn = False
        self.tol = 1e-5
        self.y_without_grad = True


class TestPrimReduceAsWithGrad3(TestPrimBaseOneGradTwoInputs):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.reduce_as_grad"
        self.dtype = "float32"
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = reduce_as_net
        self.enable_cinn = False
        self.tol = 1e-5
        self.y_without_grad = True


class TestPrimReduceAsWithGrad4(TestPrimBaseOneGradTwoInputs):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.reduce_as_grad"
        self.dtype = "float32"
        self.y_shape = [30, 1, 1]
        self.init_y_shape = [None, None, 1]
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = reduce_as_net
        self.enable_cinn = False
        self.tol = 1e-5
        self.y_without_grad = True


class TestPrimReluWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.relu_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = relu_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimRelu6WithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.relu6_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = relu6_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimReshapeWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.reshape_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = reshape_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimRollWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.roll_grad"
        self.dtype = "float32"
        self.x_shape = [100, 4, 5]
        self.init_x_shape = [None, None, 5]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = roll_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimRollWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.roll_grad"
        self.dtype = "float32"
        self.x_shape = [100, 4, 5]
        self.init_x_shape = [100, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = roll_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimScaleWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.scale_grad"
        self.dtype = "float32"
        self.x_shape = [20, 30, 70]
        self.init_x_shape = [None, None, 70]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = scale_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimScatterWithGrad(TestPrimThreeWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.scatter_grad"
        self.dtype = "float32"
        self.x_shape = [30, 50]
        self.init_x_shape = [None, 50]
        self.y_shape = [2]
        self.init_y_shape = [None]
        self.z_shape = [2, 50]
        self.init_z_shape = [None, 50]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.y = np.array([1, 2])
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = scatter_net
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        z = paddle.to_tensor(self.z, stop_gradient=False)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='float32'),
                    InputSpec(shape=self.init_z_shape, dtype='float32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y, z)
        res.backward()
        x_grad = x.gradient()
        z_grad = z.gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y, z)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [x_grad, z_grad]


class TestPrimScatterNdAddWithGrad(TestPrimScatterWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.scatter_grad"
        self.dtype = "float32"
        self.x_shape = [3, 5, 9, 10]
        self.init_x_shape = [None, None, None, 10]
        self.y_shape = [3, 2]
        self.init_y_shape = [None, 2]
        self.z_shape = [3, 9, 10]
        self.init_z_shape = [None, None, 10]
        self.x = np.ones(self.x_shape).astype(self.dtype)
        self.y = np.array([[1, 1], [0, 1], [1, 3]])
        self.z = np.random.random(self.z_shape).astype(self.dtype)
        self.net = scatter_nd_add_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSigmoidWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sigmoid_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sigmoid_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSoftmaxWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.softmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = softmax_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSoftmaxWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.softmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = softmax_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSoftmaxWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.softmax_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [30, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = softmax_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSoftsignWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.softsign_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = softsign_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.split_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.split_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.split_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSplitWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.split_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = split_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSquareWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.square_grad"
        self.dtype = "float32"
        self.x_shape = [20, 30, 70]
        self.init_x_shape = [None, None, 70]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = square_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSqueezeWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.squeeze_grad"
        self.dtype = "float32"
        self.x_shape = [1, 20, 1, 30]
        self.init_x_shape = [None, None, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = squeeze_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSqueezeWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.squeeze_grad"
        self.dtype = "float32"
        self.x_shape = [1, 20, 1, 30]
        self.init_x_shape = [None, 20, None, 30]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = squeeze_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSqueezeWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.squeeze_grad"
        self.dtype = "float32"
        self.x_shape = [1, 20, 1, 30]
        self.init_x_shape = [1, None, 1, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = squeeze_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStackWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = stack_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStackWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = stack_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStackWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = stack_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStackWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = stack_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStackWithGrad5(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        x = np.random.random(self.x_shape).astype(self.dtype)
        self.x = [x + i for i in range(4)]
        self.net = stack_net3
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = [paddle.to_tensor(self.x[i], stop_gradient=False) for i in range(4)]
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    [
                        InputSpec(shape=self.x_shape, dtype='float32'),
                        InputSpec(shape=self.init_x_shape, dtype='float32'),
                        InputSpec(shape=self.init_x_shape, dtype='float32'),
                        InputSpec(shape=self.x_shape, dtype='float32'),
                    ]
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x)
        res.backward()
        x_grad1 = x[0].gradient()
        x_grad2 = x[1].gradient()
        x_grad3 = x[2].gradient()
        x_grad4 = x[3].gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [x_grad1, x_grad2, x_grad3, x_grad4]

    def test_prim_all_dynamic(self):
        res_ref, grad_ref = self.base_net()
        res, grad = self.base_net("prim")

        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )

        for dr, d in zip(grad_ref, grad):
            np.testing.assert_allclose(dr, d, rtol=self.tol, atol=self.tol)


class TestPrimStackWithGrad6(TestPrimStackWithGrad5):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.stack_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        x = np.random.random(self.x_shape).astype(self.dtype)
        self.x = [x + i for i in range(4)]
        self.net = stack_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [1, 1, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [1, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad3(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 1]
        self.init_x_shape = [None, None, 1]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad4(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 1, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad5(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [1, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad6(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 1]
        self.init_y_shape = [None, None, 1]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad7(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad8(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.y_shape = [40]
        self.init_y_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad9(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [40]
        self.init_x_shape = [None]
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad10(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [200, 40]
        self.init_y_shape = self.y_shape
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSubtractWithGrad11(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.subtract_grad"
        self.dtype = "float32"
        self.x_shape = [200, 40]
        self.init_x_shape = self.x_shape
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = subtract_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [1000]
        self.init_x_shape = [None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSumWithGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.sum_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = sum_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSwigluWithGrad1(TestPrimBaseOneGradTwoInputs):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.swiglu_grad"
        self.dtype = "float32"
        self.y_shape = [30, 200, 40]
        self.init_y_shape = [None, None, 40]
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.random.random(self.y_shape).astype(self.dtype)
        self.net = swiglu_net1
        self.enable_cinn = False
        self.tol = 1e-5
        self.y_without_grad = True


class TestPrimSwigluWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.swiglu_grad"
        self.dtype = "float32"
        self.x_shape = [20, 30, 50, 70]
        self.init_x_shape = [None, None, None, 70]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = swiglu_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSwishWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.swish_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = swish_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTakeAlongAxisWithGrad1(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.take_along_axis_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.y_shape = [1, 1, 1]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([[[2]]], dtype="int32")
        self.net = take_along_axis_net1
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        y = paddle.broadcast_to(y, [1, 200, 40])
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='int32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y)
        res.backward()
        x_grad = x.gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [x_grad]


class TestPrimTakeAlongAxisWithGrad2(TestPrimTwoWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.take_along_axis_grad"
        self.dtype = "float32"
        self.x_shape = [2, 40, 200]
        self.init_x_shape = [None, None, None]
        self.y_shape = [2, 1, 1]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([[[1]], [[3]]], dtype="int32")
        self.net = take_along_axis_net2
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_backward_enabled(True)
        x = paddle.to_tensor(self.x, stop_gradient=False)
        y = paddle.to_tensor(self.y)
        y = paddle.broadcast_to(y, [2, 1, 200])
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                    InputSpec(shape=self.init_y_shape, dtype='int32'),
                ],
            )
            fn.train()
        else:
            fn = self.net
        res = fn(x, y)
        res.backward()
        x_grad = x.gradient()
        if flag == "prim":
            ops = [
                op.name()
                for op in fn.get_concrete_program(x, y)[-1]
                .program.backward_program.global_block()
                .ops
            ]
            assert self.op_name not in ops
            core._set_prim_backward_enabled(False)
        return res, [x_grad]


class TestPrimTakeAlongAxisWithGrad3(TestPrimTakeAlongAxisWithGrad2):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.take_along_axis_grad"
        self.dtype = "float32"
        self.x_shape = [2, 40, 200]
        self.init_x_shape = [None, None, None]
        self.y_shape = [2, 1, 1]
        self.init_y_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.y = np.array([[[1]], [[3]]], dtype="int32")
        self.net = take_along_axis_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTanhWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.tanh_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = tanh_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTileWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.tile_grad"
        self.dtype = "float32"
        self.x_shape = [10, 10, 5]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = tile_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTileWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.tile_grad"
        self.dtype = "float32"
        self.x_shape = [5, 5, 4, 3, 5, 6]
        self.init_x_shape = [None, None, None, None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = tile_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTileWithGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.tile_grad"
        self.dtype = "float32"
        self.x_shape = [5, 5, 4, 3, 2]
        self.init_x_shape = [None, None, None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = tile_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTileWithGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.tile_grad"
        self.dtype = "float32"
        self.x_shape = [5, 5, 4, 3]
        self.init_x_shape = [None, None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = tile_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTopkWithGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.topk_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, 40]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = topk_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTopkWithGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.topk_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, 200, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = topk_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTransposeWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2023)
        self.op_name = "pd_op.transpose_grad"
        self.dtype = "float32"
        self.x_shape = [20, 30, 50, 70]
        self.init_x_shape = [None, None, None, 70]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = transpose_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTruncWithGrad(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.trunc_grad"
        self.dtype = "float32"
        self.x_shape = [30, 200, 40]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = trunc_net
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad1(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net1
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad2(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net2
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad3(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net3
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad4(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net4
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad5(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad6(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net6
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad7(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [10, 20, 30]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net7
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad8(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [1]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net5
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad9(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [1]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net8
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimPNormGrad10(TestPrimBaseWithGrad):
    def setUp(self):
        np.random.seed(2024)
        self.op_name = "pd_op.p_norm_grad"
        self.dtype = "float32"
        self.x_shape = [1]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = p_norm_net9
        self.enable_cinn = False
        self.tol = 1e-6


if __name__ == "__main__":
    unittest.main()
