#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile
import unittest

import numpy as np
from dygraph_to_static_util import ast_only_test, test_and_compare_with_new_ir
from test_fetch_feed import Linear

import paddle
import paddle.nn.functional as F
from paddle import base, nn
from paddle.base import core
from paddle.nn import BatchNorm
from paddle.optimizer import Adam

np.random.seed(2020)

place = base.CUDAPlace(0) if base.is_compiled_with_cuda() else base.CPUPlace()


class PrimeNet(paddle.nn.Layer):
    def __init__(self, data_layout='NCHW'):
        super().__init__()
        self.conv = nn.Conv2D(2, 4, (3, 3), bias_attr=False)
        self.bn = BatchNorm(4, act="relu", data_layout=data_layout)

    def forward(self, x):
        y = self.conv(x)
        out = self.bn(y)
        res = F.max_pool2d(out, kernel_size=2, stride=2, padding=0)
        return res


def apply_to_static(net):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = False
    return paddle.jit.to_static(net, build_strategy=False)


def forward_post_hook_for_prim_net(layer, input, output):
    return output * 2


class TestDyToStaticSaveLoad(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(
            self.temp_dir.name, "test_dy2stat_save_load"
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_load_same_result(self):
        x_data = np.random.randn(30, 10, 32).astype('float32')
        batch_num = 3

        with base.dygraph.guard(place):
            paddle.jit.enable_to_static(True)
            x = base.dygraph.to_variable(x_data)
            net = Linear(32, 64)
            adam = Adam(learning_rate=0.1, parameters=net.parameters())

            for i in range(batch_num):
                static_out, static_loss = net(x)
                # Update parameters
                static_loss.backward()
                adam.minimize(static_loss)
                net.clear_gradients()
            # Save parameters

            paddle.save(net.state_dict(), self.model_path + '.pdparams')
            # minimize() will update parameter, call net() to get output and avg_loss.
            # Switch into eval mode.
            net.eval()
            static_out, static_loss = net(x)

        # load parameters into dygraph
        with base.dygraph.guard(place):
            dygraph_net = Linear(32, 64)

            # Load parameters
            model_dict = paddle.load(self.model_path + '.pdparams')
            dygraph_net.set_dict(model_dict)
            # Switch into eval mode.
            dygraph_net.eval()

            x = base.dygraph.to_variable(x_data)
            # predict output
            paddle.jit.enable_to_static(False)
            dygraph_out, dygraph_loss = dygraph_net(x)

        np.testing.assert_allclose(
            dygraph_out.numpy(), static_out.numpy(), rtol=1e-05
        )
        np.testing.assert_allclose(
            dygraph_loss.numpy(), static_loss.numpy(), rtol=1e-05
        )

    @ast_only_test
    @test_and_compare_with_new_ir(False)
    def test_save_load_prim(self):
        with base.dygraph.guard(place):
            self.x = paddle.randn([4, 2, 6, 6], dtype="float32")
            self.x.stop_gradient = False
            net = PrimeNet(data_layout="NCHW")
            core._set_prim_all_enabled(True)
            net.eval()
            static_net = apply_to_static(net)
            res = static_net(self.x)
            composite_program = static_net.forward.get_concrete_program(self.x)[
                1
            ].train_program
            comp_op_type_list = [
                op.type for op in composite_program.block(0).ops
            ]
            self.assertNotIn("batch_norm", comp_op_type_list)
            self.assertNotIn("relu", comp_op_type_list)
            self.assertNotIn("pow", comp_op_type_list)
            self.assertNotIn("expand_v2", comp_op_type_list)
            self.assertNotIn("unsqueeze2", comp_op_type_list)
            self.assertNotIn("reduce_mean", comp_op_type_list)
            self.assertNotIn("batch_norm_grad", comp_op_type_list)
            self.assertNotIn("relu_grad", comp_op_type_list)
            self.assertNotIn("pow_grad", comp_op_type_list)
            self.assertNotIn("expand_v2_grad", comp_op_type_list)
            self.assertNotIn("unsqueeze2_grad", comp_op_type_list)
            self.assertNotIn("reduce_mean_grad", comp_op_type_list)

            paddle.jit.save(static_net, self.model_path)
            load_func = paddle.jit.load(self.model_path)
            load_program = load_func.program()
            print("load_program:", load_program)
            load_op_type_list = [op.type for op in load_program.block(0).ops]
            new_res = load_func(self.x)
            self.assertIn("conv2d", load_op_type_list)
            self.assertIn("batch_norm", load_op_type_list)
            self.assertIn("relu", load_op_type_list)
            self.assertIn("pool2d", load_op_type_list)
            np.testing.assert_allclose(res.numpy(), new_res.numpy(), rtol=1e-05)

    @ast_only_test
    @test_and_compare_with_new_ir(False)
    def test_save_load_prim_with_hook(self):
        with base.dygraph.guard(place):
            self.x = paddle.randn([4, 2, 6, 6], dtype="float32")
            self.x.stop_gradient = False
            net = PrimeNet(data_layout="NCHW")
            net.register_forward_post_hook(forward_post_hook_for_prim_net)
            core._set_prim_all_enabled(True)
            net.eval()
            static_net = apply_to_static(net)
            res = static_net(self.x)
            composite_program = static_net.forward.get_concrete_program(self.x)[
                1
            ].train_program
            comp_op_type_list = [
                op.type for op in composite_program.block(0).ops
            ]
            self.assertNotIn("batch_norm", comp_op_type_list)
            self.assertNotIn("relu", comp_op_type_list)
            self.assertNotIn("pow", comp_op_type_list)
            self.assertNotIn("expand_v2", comp_op_type_list)
            self.assertNotIn("unsqueeze2", comp_op_type_list)
            self.assertNotIn("reduce_mean", comp_op_type_list)
            self.assertNotIn("batch_norm_grad", comp_op_type_list)
            self.assertNotIn("relu_grad", comp_op_type_list)
            self.assertNotIn("pow_grad", comp_op_type_list)
            self.assertNotIn("expand_v2_grad", comp_op_type_list)
            self.assertNotIn("unsqueeze2_grad", comp_op_type_list)
            self.assertNotIn("reduce_mean_grad", comp_op_type_list)
            self.assertNotIn("multiply_grad", comp_op_type_list)
            paddle.jit.save(static_net, self.model_path)
            load_func = paddle.jit.load(self.model_path)
            load_program = load_func.program()
            print("load_program:", load_program)
            load_op_type_list = [op.type for op in load_program.block(0).ops]
            new_res = load_func(self.x)
            self.assertIn("conv2d", load_op_type_list)
            self.assertIn("batch_norm", load_op_type_list)
            self.assertIn("relu", load_op_type_list)
            self.assertIn("pool2d", load_op_type_list)
            np.testing.assert_allclose(res.numpy(), new_res.numpy(), rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
