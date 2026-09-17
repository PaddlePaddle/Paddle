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

# [AUTO-GENERATED] Unit test for paddle.vision.ops
# Target: cover uncovered lines in paddle/python/paddle/vision/ops.py

import unittest

import paddle
from paddle.vision import ops


class TestYoloLossBasic(unittest.TestCase):
    """Test yolo_loss basic functionality.
    Tests yolo_loss (dynamically dispatched to yolo_loss function).
    """

    def setUp(self):
        paddle.disable_static()

    def test_yolov3_loss_basic(self):
        """Basic yolo_loss should return loss tensor.
        x shape: [N, C, H, W] where C = anchor_num * (class_num + 5)
        anchors is a flat list of ints.
        Returns a 1-D tensor with shape [N].
        """
        # 2 anchors, 5 classes => C = 2 * (5 + 5) = 20
        x = paddle.randn([2, 20, 13, 13], dtype='float32')
        gt_box = paddle.randn([2, 10, 4], dtype='float32')
        gt_label = paddle.randint(0, 5, [2, 10]).astype('int32')
        # anchors as flat list: [10, 13, 16, 30]
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        class_num = 5
        ignore_thresh = 0.7
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=class_num,
            ignore_thresh=ignore_thresh,
            downsample_ratio=32,
            use_label_smooth=True,
            scale_x_y=1.0,
        )
        self.assertEqual(loss.shape, [2])

    def test_yolov3_loss_with_gt_score(self):
        """yolo_loss with gt_score input."""
        # 2 anchors, 3 classes => C = 2 * (3 + 5) = 16
        x = paddle.randn([1, 16, 13, 13], dtype='float32')
        gt_box = paddle.randn([1, 5, 4], dtype='float32')
        gt_label = paddle.randint(0, 3, [1, 5]).astype('int32')
        gt_score = paddle.ones([1, 5], dtype='float32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=3,
            ignore_thresh=0.5,
            downsample_ratio=32,
            use_label_smooth=False,
            scale_x_y=1.0,
            gt_score=gt_score,
        )
        self.assertEqual(loss.shape, [1])

    def test_yolov3_loss_no_label_smooth(self):
        """yolo_loss without label smoothing."""
        # 2 anchors, 1 class => C = 2 * (1 + 5) = 12
        x = paddle.randn([1, 12, 13, 13], dtype='float32')
        gt_box = paddle.randn([1, 5, 4], dtype='float32')
        gt_label = paddle.randint(0, 1, [1, 5]).astype('int32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=1,
            ignore_thresh=0.5,
            downsample_ratio=32,
            use_label_smooth=False,
            scale_x_y=1.0,
        )
        self.assertEqual(loss.shape, [1])

    def test_yolov3_loss_float64(self):
        """yolo_loss with float64 input."""
        # 2 anchors, 1 class => C = 2 * (1 + 5) = 12
        x = paddle.randn([1, 12, 13, 13], dtype='float64')
        gt_box = paddle.randn([1, 5, 4], dtype='float64')
        gt_label = paddle.randint(0, 1, [1, 5]).astype('int32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=1,
            ignore_thresh=0.5,
            downsample_ratio=32,
            use_label_smooth=False,
            scale_x_y=1.0,
        )
        self.assertEqual(loss.dtype, paddle.float64)


class TestYoloBox(unittest.TestCase):
    """Test yolo_box basic functionality."""

    def setUp(self):
        paddle.disable_static()

    def test_yolo_box_basic(self):
        """Basic yolo_box should return boxes and scores.
        x shape: [N, C, H, W] where C = anchor_num * (5 + class_num)
        """
        # 3 anchors, 1 class => C = 3 * (5 + 1) = 18
        x = paddle.randn([1, 18, 13, 13], dtype='float32')
        img_size = paddle.to_tensor([[416, 416]], dtype='int32')
        anchors = [10, 13, 16, 30, 33, 23]
        class_num = 1
        conf_thresh = 0.01
        downsample_ratio = 32
        boxes, scores = ops.yolo_box(
            x,
            img_size,
            anchors=anchors,
            class_num=class_num,
            conf_thresh=conf_thresh,
            downsample_ratio=downsample_ratio,
        )
        # boxes: [N, M, 4], scores: [N, M, class_num]
        self.assertEqual(boxes.shape[0], 1)
        self.assertEqual(boxes.shape[2], 4)
        self.assertEqual(scores.shape[2], class_num)


if __name__ == '__main__':
    unittest.main()
