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
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# method:__getitem__||api:paddle.tensor.creation.full||method:astype||api:paddle.vision.ops.distribute_fpn_proposals||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather||method:reshape||method:transpose||method:reshape
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 256, 192, 288], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 256, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 256, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 256, 24, 36], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [1, 100, 4], dtype: paddle.float32, stop_gradient: True)
        var_5,  # (shape: [1, 100, 256], dtype: paddle.float32, stop_gradient: False)
    ):
        var_6 = var_4[0]
        var_7 = paddle.tensor.creation.full([1], 100)
        var_8 = var_7.astype('int32')
        out = paddle.vision.ops.distribute_fpn_proposals(
            var_6, 2, 5, 4, 224, rois_num=var_8
        )
        var_9 = out[0][0]
        var_10 = out[0][1]
        var_11 = out[0][2]
        var_12 = out[0][3]
        var_13 = out[1]
        var_14 = out[2][0]
        var_15 = out[2][1]
        var_16 = out[2][2]
        var_17 = out[2][3]
        var_18 = paddle.vision.ops.roi_align(
            x=var_0,
            boxes=var_9,
            boxes_num=var_14,
            output_size=7,
            spatial_scale=0.25,
            sampling_ratio=2,
            aligned=True,
        )
        var_19 = paddle.vision.ops.roi_align(
            x=var_1,
            boxes=var_10,
            boxes_num=var_15,
            output_size=7,
            spatial_scale=0.125,
            sampling_ratio=2,
            aligned=True,
        )
        var_20 = paddle.vision.ops.roi_align(
            x=var_2,
            boxes=var_11,
            boxes_num=var_16,
            output_size=7,
            spatial_scale=0.0625,
            sampling_ratio=2,
            aligned=True,
        )
        var_21 = paddle.vision.ops.roi_align(
            x=var_3,
            boxes=var_12,
            boxes_num=var_17,
            output_size=7,
            spatial_scale=0.03125,
            sampling_ratio=2,
            aligned=True,
        )
        var_22 = paddle.tensor.manipulation.concat(
            [var_18, var_19, var_20, var_21]
        )
        var_23 = paddle.tensor.manipulation.gather(var_22, var_13)
        var_24 = var_23.reshape([100, 256, -1])
        var_25 = var_24.transpose(perm=[2, 0, 1])
        var_26 = var_5.reshape([1, 100, 256])
        return var_26, var_25


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 256, 192, 288], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 96, 144], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 48, 72], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 24, 36], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 100, 256], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-8

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
