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
# 自动生成的单测，覆盖 paddle.vision.ops 模块中未覆盖的代码

"""
测试模块：paddle.vision.ops (roi_align, roi_pool, deform_conv2d, nms)
Test Module: paddle.vision.ops

本测试覆盖以下功能：
This test covers the following functions:
1. roi_align - ROI对齐 / ROI Align operation
2. roi_pool - ROI池化 / ROI Pooling operation
3. nms - 非极大值抑制 / Non-maximum suppression
4. DeformConv2D - 可变形卷积 / Deformable convolution

覆盖的未覆盖行：roi_align参数分支, roi_pool分支, deform_conv2d分支
"""

import unittest

import numpy as np

import paddle
import paddle.vision.ops as ops


class TestRoiAlign(unittest.TestCase):
    """测试ROI Align操作
    Test ROI Align operation"""

    def setUp(self):
        paddle.disable_static()

    def test_roi_align_basic(self):
        """基本ROI Align / Basic ROI Align"""
        feature_map = paddle.randn([1, 3, 8, 8])
        # boxes格式为 [x1, y1, x2, y2]，4列
        # boxes format is [x1, y1, x2, y2], 4 columns
        rois = paddle.to_tensor([[1.0, 1.0, 4.0, 4.0]], dtype='float32')
        rois_num = paddle.to_tensor([1], dtype='int32')
        out = ops.roi_align(
            feature_map, rois, output_size=3,
            spatial_scale=1.0, boxes_num=rois_num
        )
        self.assertEqual(list(out.shape), [1, 3, 3, 3])

    def test_roi_align_multiple_rois(self):
        """多个ROI / Multiple ROIs"""
        feature_map = paddle.randn([2, 3, 8, 8])
        # boxes格式为 [x1, y1, x2, y2]，通过boxes_num指定batch分配
        # boxes format is [x1, y1, x2, y2], batch assignment via boxes_num
        rois = paddle.to_tensor([
            [0.0, 0.0, 4.0, 4.0],
            [2.0, 2.0, 6.0, 6.0],
            [1.0, 1.0, 5.0, 5.0],
        ], dtype='float32')
        rois_num = paddle.to_tensor([2, 1], dtype='int32')
        out = ops.roi_align(
            feature_map, rois, output_size=2,
            spatial_scale=1.0, boxes_num=rois_num
        )
        self.assertEqual(list(out.shape), [3, 3, 2, 2])

    def test_roi_align_sampling_ratio(self):
        """指定sampling_ratio / ROI Align with sampling ratio"""
        feature_map = paddle.randn([1, 1, 16, 16])
        rois = paddle.to_tensor([[2.0, 2.0, 8.0, 8.0]], dtype='float32')
        rois_num = paddle.to_tensor([1], dtype='int32')
        out = ops.roi_align(
            feature_map, rois, output_size=4,
            spatial_scale=1.0, sampling_ratio=2,
            boxes_num=rois_num
        )
        self.assertEqual(list(out.shape), [1, 1, 4, 4])


class TestRoiPool(unittest.TestCase):
    """测试ROI Pool操作
    Test ROI Pool operation"""

    def setUp(self):
        paddle.disable_static()

    def test_roi_pool_basic(self):
        """基本ROI Pool / Basic ROI Pool"""
        feature_map = paddle.randn([1, 3, 8, 8])
        rois = paddle.to_tensor([[1.0, 1.0, 4.0, 4.0]], dtype='float32')
        rois_num = paddle.to_tensor([1], dtype='int32')
        out = ops.roi_pool(
            feature_map, rois, output_size=3,
            spatial_scale=1.0, boxes_num=rois_num
        )
        self.assertEqual(list(out.shape), [1, 3, 3, 3])


class TestNMS(unittest.TestCase):
    """测试非极大值抑制
    Test Non-Maximum Suppression"""

    def setUp(self):
        paddle.disable_static()

    def test_nms_basic(self):
        """基本NMS / Basic NMS"""
        boxes = paddle.to_tensor([
            [10.0, 10.0, 50.0, 50.0],
            [12.0, 12.0, 52.0, 52.0],
            [100.0, 100.0, 150.0, 150.0],
        ], dtype='float32')
        scores = paddle.to_tensor([0.9, 0.8, 0.7], dtype='float32')
        keep = paddle.vision.ops.nms(boxes, iou_threshold=0.5, scores=scores)
        self.assertTrue(len(keep.numpy()) >= 1)

    def test_nms_no_overlap(self):
        """无重叠NMS / NMS with no overlapping boxes"""
        boxes = paddle.to_tensor([
            [0.0, 0.0, 10.0, 10.0],
            [50.0, 50.0, 60.0, 60.0],
            [100.0, 100.0, 110.0, 110.0],
        ], dtype='float32')
        scores = paddle.to_tensor([0.9, 0.8, 0.7], dtype='float32')
        keep = paddle.vision.ops.nms(boxes, iou_threshold=0.5, scores=scores)
        self.assertEqual(len(keep.numpy()), 3)


class TestDeformConv2D(unittest.TestCase):
    """测试可变形卷积
    Test Deformable Convolution 2D"""

    def setUp(self):
        paddle.disable_static()

    def test_deform_conv2d_basic(self):
        """基本可变形卷积 / Basic deformable convolution"""
        deform_conv = paddle.vision.ops.DeformConv2D(
            in_channels=3, out_channels=8,
            kernel_size=3, padding=1
        )
        x = paddle.randn([1, 3, 8, 8])
        offset = paddle.randn([1, 18, 8, 8])
        out = deform_conv(x, offset)
        self.assertEqual(list(out.shape), [1, 8, 8, 8])

    def test_deform_conv2d_with_mask(self):
        """带mask的可变形卷积 / Deformable convolution with mask"""
        deform_conv = paddle.vision.ops.DeformConv2D(
            in_channels=3, out_channels=8,
            kernel_size=3, padding=1
        )
        x = paddle.randn([1, 3, 8, 8])
        offset = paddle.randn([1, 18, 8, 8])
        mask = paddle.ones([1, 9, 8, 8])
        out = deform_conv(x, offset, mask=mask)
        self.assertEqual(list(out.shape), [1, 8, 8, 8])


if __name__ == '__main__':
    unittest.main()
