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
# api:paddle.vision.ops.distribute_fpn_proposals||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.vision.ops.roi_align||api:paddle.tensor.manipulation.concat||api:paddle.tensor.manipulation.gather
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 256, 168, 256], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 256, 84, 128], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 256, 42, 64], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 256, 21, 32], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [512, 4], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [1], dtype: paddle.int32, stop_gradient: True)
    ):
        out = paddle.vision.ops.distribute_fpn_proposals(
            var_4, 2, 5, 4, 224, rois_num=var_5
        )
        var_6 = out[0][0]
        var_7 = out[0][1]
        var_8 = out[0][2]
        var_9 = out[0][3]
        var_10 = out[1]
        var_11 = out[2][0]
        var_12 = out[2][1]
        var_13 = out[2][2]
        var_14 = out[2][3]
        var_15 = paddle.vision.ops.roi_align(
            x=var_0,
            boxes=var_6,
            boxes_num=var_11,
            output_size=7,
            spatial_scale=0.25,
            sampling_ratio=0,
            aligned=True,
        )
        var_16 = paddle.vision.ops.roi_align(
            x=var_1,
            boxes=var_7,
            boxes_num=var_12,
            output_size=7,
            spatial_scale=0.125,
            sampling_ratio=0,
            aligned=True,
        )
        var_17 = paddle.vision.ops.roi_align(
            x=var_2,
            boxes=var_8,
            boxes_num=var_13,
            output_size=7,
            spatial_scale=0.0625,
            sampling_ratio=0,
            aligned=True,
        )
        var_18 = paddle.vision.ops.roi_align(
            x=var_3,
            boxes=var_9,
            boxes_num=var_14,
            output_size=7,
            spatial_scale=0.03125,
            sampling_ratio=0,
            aligned=True,
        )
        var_19 = paddle.tensor.manipulation.concat(
            [var_15, var_16, var_17, var_18]
        )
        var_20 = paddle.tensor.manipulation.gather(var_19, var_10)
        return var_20


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
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 256, 168, 256], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 84, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 42, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 21, 32], dtype=paddle.float32),
            paddle.rand(shape=[512, 4], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1], dtype=paddle.int32),
        )
        self.net = LayerCase
        self.with_cinn = True
        self.with_train = True

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
