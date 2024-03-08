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
# model: ppcls^configs^ImageNet^ReXNet^ReXNet_3_0
# api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.conv._conv_nd||method:squeeze||method:squeeze
import unittest

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1000, 3840, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )

    def forward(
        self,
        # (shape: [22, 3840, 1, 1], dtype: paddle.float32, stop_gradient: False)
        var_0,
    ):
        var_1 = paddle.nn.functional.common.dropout(
            var_0,
            p=0.2,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_2 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_0,
            bias=self.parameter_1,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_3 = var_2.squeeze(axis=-1)
        var_4 = var_3.squeeze(axis=-1)
        return var_4


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 3840, 1, 1], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
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
            # TODO(Aurelius84): dropout will decompose into uniform_random, which implementation
            # is different from CINN. So it's not easy to compare the result.
            pass
            # np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
