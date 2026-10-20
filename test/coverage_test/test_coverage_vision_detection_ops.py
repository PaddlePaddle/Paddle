# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

"""
图像检测函数单元测试 / Image Detection Function Unit Tests

测试目标 / Test Target:
  paddle.vision.ops 图像检测操作函数

覆盖的模块 / Covered Modules:
  - paddle.vision.ops.nms: 非最大抑制
  - paddle.vision.ops.roi_align: ROI对齐
  - paddle.vision.ops.roi_pool: ROI池化
  - paddle.vision.ops.deform_conv2d: 可变形卷积

作用 / Purpose:
  补充视觉检测相关API的测试，提升覆盖率。
"""

import unittest

import paddle

paddle.disable_static()


class TestNMS(unittest.TestCase):
    """测试非最大抑制 / Test Non-Maximum Suppression"""

    def test_nms_basic(self):
        """测试基本NMS / Test basic NMS"""
        boxes = paddle.to_tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
                [0.1, 0.1, 0.9, 0.9],  # overlaps heavily with first
            ]
        )
        scores = paddle.to_tensor([0.9, 0.6, 0.75])
        result = paddle.vision.ops.nms(boxes, iou_threshold=0.5, scores=scores)
        # Should keep first, suppress third (high overlap with first), keep second
        self.assertIsNotNone(result)
        self.assertGreater(len(result.numpy()), 0)

    def test_nms_no_overlap(self):
        """测试无重叠NMS / Test NMS with no overlap"""
        boxes = paddle.to_tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [2.0, 2.0, 3.0, 3.0],
                [4.0, 4.0, 5.0, 5.0],
            ]
        )
        scores = paddle.to_tensor([0.9, 0.8, 0.7])
        result = paddle.vision.ops.nms(boxes, iou_threshold=0.5, scores=scores)
        # All boxes should be kept
        self.assertEqual(len(result.numpy()), 3)

    def test_nms_top_k(self):
        """测试top-k NMS / Test NMS with top-k pre-filter"""
        boxes = paddle.to_tensor(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.05, 0.05, 0.95, 0.95],  # high overlap with first
                [4.0, 4.0, 5.0, 5.0],
            ]
        )
        scores = paddle.to_tensor([0.9, 0.7, 0.5])
        # top_k=2 pre-filters to top 2 scored boxes before NMS
        result = paddle.vision.ops.nms(
            boxes, iou_threshold=0.5, scores=scores, top_k=2
        )
        # Only boxes from top-2 (indices 0 and 1 by score), box 1 suppressed → result has index 0
        self.assertGreater(len(result.numpy()), 0)


class TestROIAlign(unittest.TestCase):
    """测试ROI对齐 / Test ROI Align"""

    def test_roi_align_basic(self):
        """测试基本ROI对齐 / Test basic ROI align"""
        # Feature map [batch, channels, H, W]
        x = paddle.randn([2, 8, 16, 16])
        # ROI boxes [num_boxes, 4]: [x1, y1, x2, y2]
        boxes = paddle.to_tensor(
            [
                [0.0, 0.0, 8.0, 8.0],
                [2.0, 2.0, 12.0, 12.0],
            ]
        )
        # boxes_num: number of boxes per image
        boxes_num = paddle.to_tensor([1, 1], dtype='int32')
        output = paddle.vision.ops.roi_align(
            x, boxes, boxes_num, output_size=7, spatial_scale=1.0
        )
        self.assertEqual(output.shape, [2, 8, 7, 7])

    def test_roi_align_single_batch(self):
        """测试单批次ROI对齐 / Test ROI align single batch"""
        x = paddle.randn([1, 4, 16, 16])
        boxes = paddle.to_tensor([[0.0, 0.0, 8.0, 8.0]])
        boxes_num = paddle.to_tensor([1], dtype='int32')
        output = paddle.vision.ops.roi_align(
            x, boxes, boxes_num, output_size=4, spatial_scale=1.0
        )
        self.assertEqual(output.shape, [1, 4, 4, 4])


class TestROIPool(unittest.TestCase):
    """测试ROI池化 / Test ROI Pooling"""

    def test_roi_pool_basic(self):
        """测试基本ROI池化 / Test basic ROI pooling"""
        x = paddle.randn([2, 8, 16, 16])
        boxes = paddle.to_tensor(
            [
                [0.0, 0.0, 8.0, 8.0],
                [2.0, 2.0, 12.0, 12.0],
            ]
        )
        boxes_num = paddle.to_tensor([1, 1], dtype='int32')
        output = paddle.vision.ops.roi_pool(
            x, boxes, boxes_num, output_size=7, spatial_scale=1.0
        )
        self.assertEqual(output.shape, [2, 8, 7, 7])


class TestDeformableConv(unittest.TestCase):
    """测试可变形卷积 / Test deformable convolution"""

    def test_deform_conv2d(self):
        """测试基本可变形卷积 / Test basic deformable conv2d"""
        x = paddle.randn([2, 3, 16, 16])
        kernel_size = 3
        # out_h = out_w = (16 - 3) + 1 = 14
        # Offset shape: [N, 2*k_h*k_w, out_h, out_w]
        offset = paddle.randn([2, 2 * kernel_size * kernel_size, 14, 14])
        weight = paddle.randn([8, 3, kernel_size, kernel_size])
        output = paddle.vision.ops.deform_conv2d(x, offset, weight)
        self.assertEqual(output.shape, [2, 8, 14, 14])

    def test_deform_conv2d_with_mask(self):
        """测试带掩码的可变形卷积 / Test deformable conv2d with mask"""
        x = paddle.randn([2, 3, 16, 16])
        kernel_size = 3
        offset = paddle.randn([2, 2 * kernel_size * kernel_size, 14, 14])
        mask = paddle.ones([2, kernel_size * kernel_size, 14, 14])
        weight = paddle.randn([8, 3, kernel_size, kernel_size])
        output = paddle.vision.ops.deform_conv2d(x, offset, weight, mask=mask)
        self.assertEqual(output.shape, [2, 8, 14, 14])


if __name__ == '__main__':
    unittest.main()
