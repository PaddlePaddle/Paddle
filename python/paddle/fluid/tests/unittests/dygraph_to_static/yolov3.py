#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

from darknet import ConvBNLayer, DarkNet53_conv_body

import paddle
import paddle.fluid as fluid
from paddle import _legacy_C_ops
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2Decay
from paddle.jit.api import declarative


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


#
# Training options
#
cfg = AttrDict()
# Snapshot period
cfg.snapshot_iter = 2000
# min valid area for gt boxes
cfg.gt_min_area = -1
# max target box number in an image
cfg.max_box_num = 50
# valid score threshold to include boxes
cfg.valid_thresh = 0.005
# threshold vale for box non-max suppression
cfg.nms_thresh = 0.45
# the number of top k boxes to perform nms
cfg.nms_topk = 400
# the number of output boxes after nms
cfg.nms_posk = 100
# score threshold for draw box in debug mode
cfg.draw_thresh = 0.5
# Use label smooth in class label
cfg.label_smooth = True
#
# Model options
#
# input size
cfg.input_size = 224 if sys.platform == 'darwin' else 608
# pixel mean values
cfg.pixel_means = [0.485, 0.456, 0.406]
# pixel std values
cfg.pixel_stds = [0.229, 0.224, 0.225]
# anchors box weight and height
cfg.anchors = [
    10,
    13,
    16,
    30,
    33,
    23,
    30,
    61,
    62,
    45,
    59,
    119,
    116,
    90,
    156,
    198,
    373,
    326,
]
# anchor mask of each yolo layer
cfg.anchor_masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
# IoU threshold to ignore objectness loss of pred box
cfg.ignore_thresh = 0.7
#
# SOLVER options
#
# batch size
cfg.batch_size = 1 if sys.platform == 'darwin' or os.name == 'nt' else 4
# derived learning rate the to get the final learning rate.
cfg.learning_rate = 0.001
# maximum number of iterations
cfg.max_iter = 20 if fluid.is_compiled_with_cuda() else 1
# Disable mixup in last N iter
cfg.no_mixup_iter = 10 if fluid.is_compiled_with_cuda() else 1
# warm up to learning rate
cfg.warm_up_iter = 10 if fluid.is_compiled_with_cuda() else 1
cfg.warm_up_factor = 0.0
# lr steps_with_decay
cfg.lr_steps = [400000, 450000]
cfg.lr_gamma = 0.1
# L2 regularization hyperparameter
cfg.weight_decay = 0.0005
# momentum with SGD
cfg.momentum = 0.9
#
# ENV options
#
# support both CPU and GPU
cfg.use_gpu = fluid.is_compiled_with_cuda()
# Class number
cfg.class_num = 80


class YoloDetectionBlock(fluid.dygraph.Layer):
    def __init__(self, ch_in, channel, is_test=True):
        super().__init__()

        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(
            channel
        )

        self.conv0 = ConvBNLayer(
            ch_in=ch_in,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
        )
        self.conv1 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
        )
        self.conv2 = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
        )
        self.conv3 = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
        )
        self.route = ConvBNLayer(
            ch_in=channel * 2,
            ch_out=channel,
            filter_size=1,
            stride=1,
            padding=0,
            is_test=is_test,
        )
        self.tip = ConvBNLayer(
            ch_in=channel,
            ch_out=channel * 2,
            filter_size=3,
            stride=1,
            padding=1,
            is_test=is_test,
        )

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = paddle.nn.functional.interpolate(
            x=inputs, size=out_shape, mode='nearest'
        )

        return out


class YOLOv3(fluid.dygraph.Layer):
    def __init__(self, ch_in, is_train=True, use_random=False):
        super().__init__()

        self.is_train = is_train
        self.use_random = use_random

        self.block = DarkNet53_conv_body(ch_in=ch_in, is_test=not self.is_train)
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        ch_in_list = [1024, 768, 384]
        for i in range(3):
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(
                    ch_in_list[i],
                    channel=512 // (2**i),
                    is_test=not self.is_train,
                ),
            )
            self.yolo_blocks.append(yolo_block)

            num_filters = len(cfg.anchor_masks[i]) * (cfg.class_num + 5)

            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                paddle.nn.Conv2D(
                    in_channels=1024 // (2**i),
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(
                        initializer=fluid.initializer.Normal(0.0, 0.02)
                    ),
                    bias_attr=ParamAttr(
                        initializer=fluid.initializer.Constant(0.0),
                        regularizer=L2Decay(0.0),
                    ),
                ),
            )
            self.block_outputs.append(block_out)
            if i < 2:
                route = self.add_sublayer(
                    "route2_%d" % i,
                    ConvBNLayer(
                        ch_in=512 // (2**i),
                        ch_out=256 // (2**i),
                        filter_size=1,
                        stride=1,
                        padding=0,
                        is_test=(not self.is_train),
                    ),
                )
                self.route_blocks_2.append(route)
            self.upsample = Upsample()

    @declarative
    def forward(
        self,
        inputs,
        gtbox=None,
        gtlabel=None,
        gtscore=None,
        im_id=None,
        im_shape=None,
    ):
        self.outputs = []
        self.boxes = []
        self.scores = []
        self.losses = []
        self.downsample = 32
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                block = fluid.layers.concat(
                    input=[route, block], axis=1  # noqa: F821
                )
            route, tip = self.yolo_blocks[i](block)
            block_out = self.block_outputs[i](tip)
            self.outputs.append(block_out)

            if i < 2:
                route = self.route_blocks_2[i](route)
                route = self.upsample(route)
        self.gtbox = gtbox
        self.gtlabel = gtlabel
        self.gtscore = gtscore
        self.im_id = im_id
        self.im_shape = im_shape

        # cal loss
        for i, out in enumerate(self.outputs):
            anchor_mask = cfg.anchor_masks[i]
            if self.is_train:
                loss = paddle.vision.ops.yolo_loss(
                    x=out,
                    gt_box=self.gtbox,
                    gt_label=self.gtlabel,
                    gt_score=self.gtscore,
                    anchors=cfg.anchors,
                    anchor_mask=anchor_mask,
                    class_num=cfg.class_num,
                    ignore_thresh=cfg.ignore_thresh,
                    downsample_ratio=self.downsample,
                    use_label_smooth=cfg.label_smooth,
                )
                self.losses.append(paddle.mean(loss))

            else:
                mask_anchors = []
                for m in anchor_mask:
                    mask_anchors.append(cfg.anchors[2 * m])
                    mask_anchors.append(cfg.anchors[2 * m + 1])
                boxes, scores = paddle.vision.ops.yolo_box(
                    x=out,
                    img_size=self.im_shape,
                    anchors=mask_anchors,
                    class_num=cfg.class_num,
                    conf_thresh=cfg.valid_thresh,
                    downsample_ratio=self.downsample,
                    name="yolo_box" + str(i),
                )
                self.boxes.append(boxes)
                self.scores.append(paddle.transpose(scores, perm=[0, 2, 1]))
            self.downsample //= 2

        if not self.is_train:
            # get pred
            yolo_boxes = fluid.layers.concat(self.boxes, axis=1)
            yolo_scores = fluid.layers.concat(self.scores, axis=2)

            pred = _legacy_C_ops.multiclass_nms(
                bboxes=yolo_boxes,
                scores=yolo_scores,
                score_threshold=cfg.valid_thresh,
                nms_top_k=cfg.nms_topk,
                keep_top_k=cfg.nms_posk,
                nms_threshold=cfg.nms_thresh,
                background_label=-1,
            )
            return pred
        else:
            return sum(self.losses)
