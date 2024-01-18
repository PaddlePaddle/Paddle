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

# repo: PaddleDetection
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.manipulation.split
import unittest

import numpy as np

import paddle


class SIR183(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_739 = self.create_parameter(
            shape=[112],
            dtype=paddle.float32,
        )
        self.var_738 = self.create_parameter(
            shape=[112, 96, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_735,  # (shape: [1, 96, 40, 40], dtype: paddle.float32, stop_gradient: False)
    ):
        var_740 = paddle.nn.functional.conv._conv_nd(
            var_735,
            self.var_738,
            bias=self.var_739,
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
        out = paddle.tensor.manipulation.split(var_740, [80, 32], axis=1)
        var_741 = out[0]
        var_742 = out[1]
        return var_740, var_741, var_742


class TestSIR183(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 96, 40, 40], dtype=paddle.float32),
        )
        self.net = SIR183()

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
