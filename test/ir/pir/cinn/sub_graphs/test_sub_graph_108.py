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
# model: configs^ppyoloe^ppyoloe_crn_l_300e_coco_single_dy2st_train
# api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,method||__getitem__,method||reshape,api||paddle.nn.functional.activation.softmax,method||transpose,api||paddle.nn.functional.conv._conv_nd,method||squeeze,api||paddle.tensor.manipulation.split,method||__neg__,method||__add__,method||__add__,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR170(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_927 = self.create_parameter(
            shape=[1, 17, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_918,  # (shape: [4116, 2], dtype: paddle.float32, stop_gradient: True)
        var_919,  # (shape: [1, 4116, 68], dtype: paddle.float32, stop_gradient: False)
    ):
        var_920 = paddle.tensor.attribute.shape(var_919)
        var_921 = var_920.__getitem__(0)
        var_922 = var_920.__getitem__(1)
        var_923 = var_920.__getitem__(2)
        var_924 = var_919.reshape([-1, var_922, 4, 17])
        var_925 = paddle.nn.functional.activation.softmax(var_924)
        var_926 = var_925.transpose([0, 3, 1, 2])
        var_928 = paddle.nn.functional.conv._conv_nd(
            var_926,
            self.var_927,
            bias=None,
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
        var_929 = var_928.squeeze(1)
        out = paddle.tensor.manipulation.split(var_929, 2, -1)
        var_930 = out[0]
        var_931 = out[1]
        var_932 = var_930.__neg__()
        var_933 = var_932.__add__(var_918)
        var_934 = var_931.__add__(var_918)
        var_935 = paddle.tensor.manipulation.concat([var_933, var_934], -1)
        return var_935


class TestSIR170(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[4116, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 4116, 68], dtype=paddle.float32),
        )
        self.net = SIR170()

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
