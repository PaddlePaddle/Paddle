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

# Copied from `base.py` and `test_sub_graph_4.py` under `test/ir/pir/cinn/sub_graphs/`

import unittest

import numpy as np

import paddle
from paddle.static import InputSpec


class TestBase(unittest.TestCase):
    def setUp(self):
        # default setting
        self.net = None
        self.inputs = None
        self.input_specs = None
        self.with_prim = True
        self.with_cinn = True
        self.atol = 1e-6
        self.train_atol = 1e-6
        self.with_precision_compare = True
        self.with_train = True  # 本个pr中默认为false，下个增量pr中改为默认true
        # override customized settting
        self.init()
        if self.inputs:
            self.set_input_grad()

    def set_input_grad(self):
        if self.with_train:
            for i in range(len(self.inputs)):
                self.inputs[i].stop_gradient = False

    def init(self):
        pass

    def set_flags(self):
        pass

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.seed(123)
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net(),
                    build_strategy=build_strategy,
                    full_graph=True,
                    input_spec=self.input_specs,
                )
            else:
                net = paddle.jit.to_static(
                    net(), full_graph=True, input_spec=self.input_specs
                )
        if self.with_train:
            net.train()
        else:
            net.eval()
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        if not self.net:
            return
        st_out = self.train(self.net, to_static=True)
        self.set_flags()
        cinn_out = self.train(
            self.net,
            to_static=True,
            with_prim=self.with_prim,
            with_cinn=self.with_cinn,
        )
        if self.with_precision_compare:
            for st, cinn in zip(
                paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
            ):
                np.testing.assert_allclose(
                    st.numpy(), cinn.numpy(), atol=self.atol
                )
        if self.with_train:
            st_loss = st_out.mean()
            st_loss.backward()
            st_grad = []
            for i in range(len(self.inputs)):
                if self.inputs[i].dtype != paddle.int64:
                    st_grad.append(self.inputs[i].grad.numpy().copy())
            cinn_loss = cinn_out.mean()
            cinn_loss.backward()
            cinn_grad = []
            for i in range(len(self.inputs)):
                if self.inputs[i].dtype != paddle.int64:
                    cinn_grad.append(self.inputs[i].grad.numpy().copy())
            for i in range(len(cinn_grad)):
                np.testing.assert_allclose(
                    st_grad[i], cinn_grad[i], atol=self.train_atol
                )


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[128, 640],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 196, 128], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.common.linear(
            x=var_0, weight=self.parameter_0, bias=None, name=None
        )
        var_2 = var_1.flatten(0, 1)
        return var_2, var_1


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[22, 196, 128], dtype=paddle.float32),)
        self.net = LayerCase
        self.with_train = False


if __name__ == '__main__':
    unittest.main()
