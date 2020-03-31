# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay

from model import Model, Loss
from .darknet import DarkNet53, ConvBNLayer

__all__ = ['YoloLoss', 'YOLOv3']


class YoloDetectionBlock(fluid.dygraph.Layer):
    def __init__(self, ch_in, channel):
        super(YoloDetectionBlock, self).__init__()

        assert channel % 2 == 0, \
            "channel {} cannot be divided by 2".format(channel)

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)
        self.conv1 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1)
        self.conv2 = ConvBNLayer(
            ch_in=channel*2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)
        self.conv3 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1)
        self.route = ConvBNLayer(
            ch_in=channel*2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0)
        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel*2,
            filter_size=3,
            stride=1,
            padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class YOLOv3(Model):
    def __init__(self, num_classes=80, model_mode='train'):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        assert str.lower(model_mode) in ['train', 'eval'], \
            "model_mode should be 'train' or 'val', but got " \
            "{}".format(model_mode)
        self.model_mode = str.lower(model_mode)
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,
                        59, 119, 116, 90, 156, 198, 373, 326]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        self.valid_thresh = 0.005
        self.nms_thresh = 0.45
        self.nms_topk = 400
        self.nms_posk = 100
        self.draw_thresh = 0.5

        self.block = DarkNet53()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks = []

        for idx, num_chan in enumerate([1024, 768, 384]):
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_{}".format(idx),
                YoloDetectionBlock(num_chan, 512 // (2**idx)))
            self.yolo_blocks.append(yolo_block)

            num_filters = len(self.anchor_masks[idx]) * (self.num_classes + 5)

            block_out = self.add_sublayer(
                "block_out_{}".format(idx),
                Conv2D(num_channels=1024 // (2**idx),
                       num_filters=num_filters,
                       filter_size=1,
                       act=None,
                       param_attr=ParamAttr(
                           initializer=fluid.initializer.Normal(0., 0.02)),
                       bias_attr=ParamAttr(
                           initializer=fluid.initializer.Constant(0.0),
                           regularizer=L2Decay(0.))))
            self.block_outputs.append(block_out)
            if idx < 2:
                route = self.add_sublayer(
                    "route2_{}".format(idx),
                    ConvBNLayer(ch_in=512 // (2**idx),
                                ch_out=256 // (2**idx),
                                filter_size=1,
                                act='leaky_relu'))
                self.route_blocks.append(route)

    def forward(self, img_info, inputs):
        outputs = []
        boxes = []
        scores = []
        downsample = 32

        feats = self.block(inputs)
        route = None
        for idx, feat in enumerate(feats):
            if idx > 0:
                feat = fluid.layers.concat(input=[route, feat], axis=1)
            route, tip = self.yolo_blocks[idx](feat)
            block_out = self.block_outputs[idx](tip)
            outputs.append(block_out)

            if idx < 2:
                route = self.route_blocks[idx](route)
                route = fluid.layers.resize_nearest(route, scale=2)

            if self.model_mode == 'eval':
                anchor_mask = self.anchor_masks[idx]
                mask_anchors = []
                for m in anchor_mask:
                    mask_anchors.append(self.anchors[2 * m])
                    mask_anchors.append(self.anchors[2 * m + 1])
                img_shape = fluid.layers.slice(img_info, axes=[1], starts=[1], ends=[3])
                img_id = fluid.layers.slice(img_info, axes=[1], starts=[0], ends=[1])
                b, s = fluid.layers.yolo_box(
                    x=block_out,
                    img_size=img_shape,
                    anchors=mask_anchors,
                    class_num=self.num_classes,
                    conf_thresh=self.valid_thresh,
                    downsample_ratio=downsample)

                boxes.append(b)
                scores.append(fluid.layers.transpose(s, perm=[0, 2, 1]))

            downsample //= 2

        if self.model_mode == 'train':
            return outputs

        return outputs + [img_id[0, :], fluid.layers.multiclass_nms(
                bboxes=fluid.layers.concat(boxes, axis=1),
                scores=fluid.layers.concat(scores, axis=2),
                score_threshold=self.valid_thresh,
                nms_top_k=self.nms_topk,
                keep_top_k=self.nms_posk,
                nms_threshold=self.nms_thresh,
                background_label=-1)
]


class YoloLoss(Loss):
    def __init__(self, num_classes=80, num_max_boxes=50):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.num_max_boxes = num_max_boxes
        self.ignore_thresh = 0.7
        self.anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45,
                        59, 119, 116, 90, 156, 198, 373, 326]
        self.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    def forward(self, outputs, labels):
        downsample = 32
        gt_box, gt_label, gt_score = labels
        losses = []

        for idx, out in enumerate(outputs):
            if idx == 3: break # debug
            anchor_mask = self.anchor_masks[idx]
            loss = fluid.layers.yolov3_loss(
                x=out,
                gt_box=gt_box,
                gt_label=gt_label,
                gt_score=gt_score,
                anchor_mask=anchor_mask,
                downsample_ratio=downsample,
                anchors=self.anchors,
                class_num=self.num_classes,
                ignore_thresh=self.ignore_thresh,
                use_label_smooth=True)
            loss = fluid.layers.reduce_mean(loss)
            losses.append(loss)
            downsample //= 2
        return losses
