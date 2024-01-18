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
# model: configs^cascade_rcnn^cascade_rcnn_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.vision.ops.distribute_fpn_proposals,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.vision.ops.roi_align,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.gather
import unittest

import numpy as np

import paddle


class SIR87(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1647,  # (shape: [1, 256, 192, 288], dtype: paddle.float32, stop_gradient: False)
        var_1648,  # (shape: [1, 256, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_1649,  # (shape: [1, 256, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_1650,  # (shape: [1, 256, 24, 36], dtype: paddle.float32, stop_gradient: False)
        var_1652,  # (shape: [512, 4], dtype: paddle.float32, stop_gradient: False)
        var_1653,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
    ):
        out = paddle.vision.ops.distribute_fpn_proposals(
            var_1652, 2, 5, 4, 224, rois_num=var_1653
        )
        var_1654 = out[0][0]
        var_1655 = out[0][1]
        var_1656 = out[0][2]
        var_1657 = out[0][3]
        var_1658 = out[1]
        var_1659 = out[2][0]
        var_1660 = out[2][1]
        var_1661 = out[2][2]
        var_1662 = out[2][3]
        var_1663 = paddle.vision.ops.roi_align(
            x=var_1647,
            boxes=var_1654,
            boxes_num=var_1659,
            output_size=7,
            spatial_scale=0.25,
            sampling_ratio=0,
            aligned=True,
        )
        var_1664 = paddle.vision.ops.roi_align(
            x=var_1648,
            boxes=var_1655,
            boxes_num=var_1660,
            output_size=7,
            spatial_scale=0.125,
            sampling_ratio=0,
            aligned=True,
        )
        var_1665 = paddle.vision.ops.roi_align(
            x=var_1649,
            boxes=var_1656,
            boxes_num=var_1661,
            output_size=7,
            spatial_scale=0.0625,
            sampling_ratio=0,
            aligned=True,
        )
        var_1666 = paddle.vision.ops.roi_align(
            x=var_1650,
            boxes=var_1657,
            boxes_num=var_1662,
            output_size=7,
            spatial_scale=0.03125,
            sampling_ratio=0,
            aligned=True,
        )
        var_1667 = paddle.tensor.manipulation.concat(
            [var_1663, var_1664, var_1665, var_1666]
        )
        var_1668 = paddle.tensor.manipulation.gather(var_1667, var_1658)
        return var_1668


class TestSIR87(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 192, 288], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 96, 144], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 48, 72], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 24, 36], dtype=paddle.float32),
            paddle.rand(shape=[512, 4], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        )
        self.net = SIR87()

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
