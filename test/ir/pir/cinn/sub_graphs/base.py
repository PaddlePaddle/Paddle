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

import paddle


class TestBase(unittest.TestCase):
    def setUp(self):
        # default setting
        self.net = None
        self.inputs = None
        self.input_specs = None
        self.with_prim = True
        self.with_cinn = True
        self.atol = 1e-6
        # override customized settting
        self.init()

    def init(self):
        pass

    def set_flags(self):
        pass

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net,
                    build_strategy=build_strategy,
                    full_graph=True,
                    input_spec=self.input_specs,
                )
            else:
                net = paddle.jit.to_static(
                    net, full_graph=True, input_spec=self.input_specs
                )
        paddle.seed(123)
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
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)
