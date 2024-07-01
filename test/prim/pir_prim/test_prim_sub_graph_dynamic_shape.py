# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.framework import core
from paddle.static import InputSpec


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


def log_softmax_net(x):
    return F.log_softmax(x)


def any_net(x):
    return paddle.any(x)


def embedding_net(x):
    w = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
            [27, 28, 29],
        ],
        dtype=np.float32,
    )
    w = paddle.to_tensor(w)
    return F.embedding(x, w, padding_idx=1)


def full_like_net(x):
    return paddle.full_like(x, 1)


def stack_net(x):
    y = x + 1
    return paddle.stack([x, y], axis=0)


def clip_net(x):
    return paddle.clip(x, 0, 1)


def index_sample_net(x, index):
    return paddle.index_sample(x, index)


def huber_loss_net(x, label):
    return paddle._C_ops.huber_loss(x, label, 1.0)


def bce_loss_net(x, label):
    return paddle._C_ops.bce_loss(x, label)


def swiglu_net1(x, y):
    return paddle.incubate.nn.functional.swiglu(x, y)


def swiglu_net2(x):
    return paddle.incubate.nn.functional.swiglu(x)


def squared_l2_norm_net(x):
    return paddle._C_ops.squared_l2_norm(x)


def elu_net(x):
    return paddle.nn.functional.elu(x, 1.0)


def dropout_net1(x):
    return paddle.nn.functional.dropout(x, 0.5)


def mean_all_net1(x):
    return paddle._C_ops.mean_all(x)


group_norm1 = paddle.nn.GroupNorm(num_channels=128, num_groups=32)


def group_norm_net1(x):
    return group_norm1(x)


group_norm2 = paddle.nn.GroupNorm(
    num_channels=128, num_groups=32, weight_attr=False
)


def group_norm_net2(x):
    return group_norm2(x)


group_norm3 = paddle.nn.GroupNorm(
    num_channels=128, num_groups=32, bias_attr=False
)


def group_norm_net3(x):
    return group_norm3(x)


group_norm4 = paddle.nn.GroupNorm(
    num_channels=128,
    num_groups=32,
    weight_attr=False,
    bias_attr=False,
)


def group_norm_net4(x):
    return group_norm4(x)


group_norm5 = paddle.nn.GroupNorm(
    num_channels=128,
    num_groups=32,
    weight_attr=False,
    bias_attr=False,
    data_format='NHWC',
)


def group_norm_net5(x):
    return group_norm5(x)


def layer_norm_net1(x):
    return paddle.nn.functional.layer_norm(x, x.shape[1:])


def instance_norm_net(x):
    return paddle.nn.functional.instance_norm(x)


def flatten_net(x):
    return paddle.flatten(x, 1, 2)


def meshgrid_net(x, y):
    return paddle.meshgrid(x, y)


def unbind_net(x):
    return paddle.unbind(x)


class TestPrimBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, 4096]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = log_softmax_net
        self.necessary_ops = "pd_op.log_softmax"
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype='float32'),
                ],
            )
            fn.eval()
        else:
            fn = self.net
        res = fn(x)

        if flag == "prim":
            ops = [
                op.name()
                for op in fn.program_cache.last()[-1][-1]
                .infer_program.program.global_block()
                .ops
            ]
            assert self.necessary_ops not in ops
            core._set_prim_all_enabled(False)
        return res

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(
                ref, actual, rtol=self.tol, atol=self.tol
            )


class TestPrimAny(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "bool"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, 4096]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = any_net
        self.necessary_ops = "pd_op.any"
        self.enable_cinn = False
        self.tol = 1e-6


class TestEmbedding(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "int"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, 4096]
        self.x = np.random.randint(0, 10, size=self.x_shape)
        self.net = embedding_net
        self.necessary_ops = "pd_op.embedding"
        self.enable_cinn = False
        self.tol = 1e-6


class TestUnbind(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [4, 5, 6]
        self.init_x_shape = [4, 5, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = unbind_net
        self.necessary_ops = "pd_op.unbind"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimFullLike(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, 4096]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = full_like_net
        self.necessary_ops = "pd_op.full_like"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimStack(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, 4096]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = stack_net
        self.necessary_ops = "pd_op.stack"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimClip(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [1, 300, 4096]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = clip_net
        self.necessary_ops = "pd_op.clip"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimClip2(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = []
        self.init_x_shape = []
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = clip_net
        self.necessary_ops = "pd_op.clip"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSquaredL2Norm(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 5, 10]
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = squared_l2_norm_net
        self.necessary_ops = "pd_op.squared_l2_norm"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimElu(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [15, 20]
        self.init_x_shape = [None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = elu_net
        self.necessary_ops = "pd_op.elu"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimTwo(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300, 4096]
        self.shape_y = [300, 2048]
        self.dtype_x = "float32"
        self.dtype_y = int
        self.init_x_shape = [None, 4096]
        self.init_y_shape = [None, 2048]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.y = np.random.random(self.shape_y).astype(self.dtype_y)
        self.net = index_sample_net
        self.necessary_ops = "pd_op.index_sample"
        self.enable_cinn = False
        self.tol = 1e-6

    def base_net(self, flag=None):
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        if flag == "prim":
            core._set_prim_all_enabled(True)
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=self.init_x_shape, dtype=self.dtype_x),
                    InputSpec(shape=self.init_y_shape, dtype=self.dtype_y),
                ],
            )
            fn.eval()
        else:
            fn = self.net
        res = fn(x, y)

        if flag == "prim":
            ops = [
                op.name()
                for op in fn.program_cache.last()[-1][-1]
                .infer_program.program.global_block()
                .ops
            ]
            assert self.necessary_ops not in ops
            core._set_prim_all_enabled(False)
        return res

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=self.tol)


class TestPrimTwoIndexSample(TestPrimTwo):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300, 4096]
        self.shape_y = [300, 2048]
        self.dtype_x = "float32"
        self.dtype_y = int
        self.init_x_shape = [None, 4096]
        self.init_y_shape = [300, 2048]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.y = np.random.random(self.shape_y).astype(self.dtype_y)
        self.net = index_sample_net
        self.necessary_ops = "pd_op.index_sample"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimHuberLoss(TestPrimTwo):
    def setUp(self):
        np.random.seed(2023)
        self.x_shape = [100, 1]
        self.y_shape = [100, 1]
        self.dtype_x = "float32"
        self.dtype_y = "float32"
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.uniform(0, 1.0, self.x_shape).astype(self.dtype_x)
        self.y = np.random.uniform(0, 1.0, self.y_shape).astype(self.dtype_y)
        self.net = huber_loss_net
        self.necessary_ops = "pd_op.huber_loss"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimBceLoss(TestPrimTwo):
    def setUp(self):
        np.random.seed(2023)
        self.x_shape = [20, 30, 40, 50]
        self.y_shape = [20, 30, 40, 50]
        self.dtype_x = "float32"
        self.dtype_y = "float32"
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.uniform(0.1, 0.8, self.x_shape).astype(self.dtype_x)
        self.y = np.random.randint(0, 2, self.x_shape).astype(self.dtype_y)
        self.net = bce_loss_net
        self.necessary_ops = "pd_op.bce_loss"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSwiglu1(TestPrimTwo):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300, 4096]
        self.shape_y = [300, 4096]
        self.dtype_x = "float32"
        self.dtype_y = "float32"
        self.init_x_shape = [None, None]
        self.init_y_shape = [None, None]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.y = np.random.random(self.shape_y).astype(self.dtype_y)
        self.net = swiglu_net1
        self.necessary_ops = "pd_op.swiglu"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimSwiglu2(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300, 4096]
        self.dtype_x = "float32"
        self.init_x_shape = [None, 4096]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.net = swiglu_net2
        self.necessary_ops = "pd_op.swiglu"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimLayernorm(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 32, 128]
        self.dtype_x = "float32"
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.net = layer_norm_net1
        self.necessary_ops = "pd_op.layer_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimInstancenorm(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [2, 32, 128]
        self.dtype_x = "float32"
        self.init_x_shape = [None, None, None]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.net = instance_norm_net
        self.necessary_ops = "pd_op.instance_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm1(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 128, 10, 20]
        self.init_x_shape = [None, 128, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net1
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm2(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 128, 10, 20]
        self.init_x_shape = [None, 128, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net2
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm3(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [50, 128, 10]
        self.init_x_shape = [None, 128, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net3
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm4(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 128, 10, 20]
        self.init_x_shape = [None, 128, None, None]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net4
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm5(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 6, 8, 4, 128]
        self.init_x_shape = [8, 6, 8, 4, 128]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net5
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm6(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 6, 8, 4, 128]
        self.init_x_shape = [None, None, None, None, 128]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net5
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimGroupNorm7(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.x_shape = [8, 10, 8, 128]
        self.init_x_shape = [None, None, None, 128]
        self.x = np.random.random(self.x_shape).astype(self.dtype)
        self.net = group_norm_net5
        self.necessary_ops = "pd_op.group_norm"
        self.enable_cinn = False
        self.tol = 5e-6


class TestPrimDropout(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        paddle.seed(2023)
        self.shape_x = [300, 4096]
        self.dtype_x = "float32"
        self.init_x_shape = [None, 4096]
        self.x = np.ones(self.shape_x).astype(self.dtype_x)
        self.net = dropout_net1
        self.necessary_ops = "pd_op.dropout"
        self.enable_cinn = False

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref.sum(), actual.sum(), rtol=0.08)


class TestPrimMeshgrid(TestPrimTwo):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [300]
        self.shape_y = []
        self.dtype_x = "float32"
        self.dtype_y = "float32"
        self.init_x_shape = [None]
        self.init_y_shape = [None]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.y = np.random.random(self.shape_y).astype(self.dtype_y)
        self.net = meshgrid_net
        self.necessary_ops = "pd_op.meshgrid"
        self.enable_cinn = False
        self.tol = 1e-6


class TestPrimMeanAll(TestPrimBase):
    def setUp(self):
        np.random.seed(2023)
        paddle.seed(2023)
        self.shape_x = [300, 4096]
        self.dtype_x = "float32"
        self.init_x_shape = [None, 4096]
        self.x = np.random.random(self.shape_x).astype(self.dtype_x)
        self.net = mean_all_net1
        self.necessary_ops = "pd_op.mean_all"
        self.enable_cinn = False


if __name__ == "__main__":
    unittest.main()
