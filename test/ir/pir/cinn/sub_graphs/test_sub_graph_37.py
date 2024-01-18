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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^DeiT^DeiT_tiny_distilled_patch16_224
# api||paddle.nn.functional.norm.layer_norm,api||paddle.nn.functional.common.linear,method||reshape,method||transpose,method||__getitem__,method||__getitem__,method||__getitem__,method||transpose,method||matmul,method||__mul__,api||paddle.nn.functional.activation.softmax,api||paddle.nn.functional.common.dropout,method||matmul,method||transpose,method||reshape,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.common.dropout
import unittest

import numpy as np

import paddle


class SIR4(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_238 = self.create_parameter(
            shape=[192, 192],
            dtype=paddle.float32,
        )
        self.var_219 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_239 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_222 = self.create_parameter(
            shape=[192, 576],
            dtype=paddle.float32,
        )
        self.var_220 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.var_223 = self.create_parameter(
            shape=[576],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_218,  # (shape: [86, 198, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_221 = paddle.nn.functional.norm.layer_norm(
            var_218,
            normalized_shape=[192],
            weight=self.var_219,
            bias=self.var_220,
            epsilon=1e-06,
        )
        var_224 = paddle.nn.functional.common.linear(
            x=var_221, weight=self.var_222, bias=self.var_223, name=None
        )
        var_225 = var_224.reshape((-1, 198, 3, 3, 64))
        var_226 = var_225.transpose((2, 0, 3, 1, 4))
        var_227 = var_226.__getitem__(0)
        var_228 = var_226.__getitem__(1)
        var_229 = var_226.__getitem__(2)
        var_230 = var_228.transpose((0, 1, 3, 2))
        var_231 = var_227.matmul(var_230)
        var_232 = var_231.__mul__(0.125)
        var_233 = paddle.nn.functional.activation.softmax(var_232, axis=-1)
        var_234 = paddle.nn.functional.common.dropout(
            var_233,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_235 = var_234.matmul(var_229)
        var_236 = var_235.transpose((0, 2, 1, 3))
        var_237 = var_236.reshape((-1, 198, 192))
        var_240 = paddle.nn.functional.common.linear(
            x=var_237, weight=self.var_238, bias=self.var_239, name=None
        )
        var_241 = paddle.nn.functional.common.dropout(
            var_240,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        return var_241


class TestSIR4(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[86, 198, 192], dtype=paddle.float32),)
        self.net = SIR4()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
