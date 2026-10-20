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

import unittest

import numpy as np

import paddle
from paddle.vision import ops


class TestYoloLoss(unittest.TestCase):
    """测试 yolo_loss / Test yolo_loss in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_yolo_loss_basic(self):
        """基本yolo_loss测试 / Basic yolo_loss test.
        C = anchor_num * (class_num + 5).
        """
        # 2 anchors, 2 classes => C = 2 * (2 + 5) = 14
        x = paddle.randn([2, 14, 8, 8], dtype='float32')
        gt_box = paddle.randn([2, 10, 4], dtype='float32')
        gt_label = paddle.randint(0, 2, [2, 10]).astype('int32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=8,
            use_label_smooth=True,
            scale_x_y=1.0,
        )
        self.assertEqual(loss.shape, [2])
        # 损失值应为有限数值 / Loss should be finite
        self.assertTrue(np.all(np.isfinite(loss.numpy())))

    def test_yolo_loss_with_gt_score(self):
        """带gt_score的yolo_loss测试 / yolo_loss with gt_score."""
        x = paddle.randn([1, 14, 8, 8], dtype='float32')
        gt_box = paddle.randn([1, 5, 4], dtype='float32')
        gt_label = paddle.randint(0, 2, [1, 5]).astype('int32')
        gt_score = paddle.ones([1, 5], dtype='float32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=2,
            ignore_thresh=0.5,
            downsample_ratio=16,
            use_label_smooth=False,
            scale_x_y=1.0,
            gt_score=gt_score,
        )
        self.assertEqual(loss.shape, [1])

    def test_yolo_loss_scale_x_y(self):
        """测试不同scale_x_y参数 / Test different scale_x_y."""
        x = paddle.randn([2, 14, 8, 8], dtype='float32')
        gt_box = paddle.randn([2, 10, 4], dtype='float32')
        gt_label = paddle.randint(0, 2, [2, 10]).astype('int32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=32,
            scale_x_y=1.2,
        )
        self.assertEqual(loss.shape, [2])

    def test_yolo_loss_float64(self):
        """测试float64输入 / Test float64 input."""
        x = paddle.randn([1, 14, 4, 4], dtype='float64')
        gt_box = paddle.randn([1, 3, 4], dtype='float64')
        gt_label = paddle.randint(0, 2, [1, 3]).astype('int32')
        anchors = [10, 13, 16, 30]
        anchor_mask = [0, 1]
        loss = ops.yolo_loss(
            x,
            gt_box,
            gt_label,
            anchors=anchors,
            anchor_mask=anchor_mask,
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=8,
        )
        self.assertEqual(loss.dtype, paddle.float64)
        self.assertEqual(loss.shape, [1])


class TestYoloBox(unittest.TestCase):
    """测试 yolo_box / Test yolo_box in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_yolo_box_basic(self):
        """基本yolo_box测试 / Basic yolo_box test.
        C = anchor_num * (5 + class_num).
        """
        # 2 anchors, 2 classes => C = 2 * (5 + 2) = 14
        x = paddle.randn([2, 14, 8, 8], dtype='float32')
        img_size = paddle.ones((2, 2)).astype('int32') * 64
        anchors = [10, 13, 16, 30]
        boxes, scores = ops.yolo_box(
            x,
            img_size=img_size,
            anchors=anchors,
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.0,
        )
        # boxes shape: [N, M, 4], scores shape: [N, M, class_num]
        self.assertEqual(len(boxes.shape), 3)
        self.assertEqual(boxes.shape[2], 4)
        self.assertEqual(scores.shape[2], 2)

    def test_yolo_box_iou_aware(self):
        """测试iou_aware模式 / Test iou_aware mode.
        When iou_aware=True, C = anchor_num * (6 + class_num).
        """
        # 2 anchors, 2 classes, iou_aware => C = 2 * (6 + 2) = 16
        x = paddle.randn([2, 16, 8, 8], dtype='float32')
        img_size = paddle.ones((2, 2)).astype('int32') * 64
        anchors = [10, 13, 16, 30]
        boxes, scores = ops.yolo_box(
            x,
            img_size=img_size,
            anchors=anchors,
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            clip_bbox=True,
            scale_x_y=1.0,
            iou_aware=True,
            iou_aware_factor=0.5,
        )
        self.assertEqual(len(boxes.shape), 3)
        self.assertEqual(boxes.shape[2], 4)

    def test_yolo_box_no_clip(self):
        """测试不裁剪bbox / Test without clipping bbox."""
        x = paddle.randn([1, 14, 8, 8], dtype='float32')
        img_size = paddle.ones((1, 2)).astype('int32') * 64
        anchors = [10, 13, 16, 30]
        boxes, scores = ops.yolo_box(
            x,
            img_size=img_size,
            anchors=anchors,
            class_num=2,
            conf_thresh=0.5,
            downsample_ratio=8,
            clip_bbox=False,
        )
        self.assertEqual(len(boxes.shape), 3)

    def test_yolo_box_scale_x_y(self):
        """测试scale_x_y参数 / Test scale_x_y parameter."""
        x = paddle.randn([1, 14, 8, 8], dtype='float32')
        img_size = paddle.ones((1, 2)).astype('int32') * 64
        anchors = [10, 13, 16, 30]
        boxes, scores = ops.yolo_box(
            x,
            img_size=img_size,
            anchors=anchors,
            class_num=2,
            conf_thresh=0.01,
            downsample_ratio=8,
            scale_x_y=1.5,
        )
        self.assertEqual(len(boxes.shape), 3)


class TestPriorBox(unittest.TestCase):
    """测试 prior_box / Test prior_box in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_prior_box_basic(self):
        """基本prior_box测试 / Basic prior_box test."""
        inp = paddle.randn((1, 3, 6, 9), dtype=paddle.float32)
        image = paddle.randn((1, 3, 9, 12), dtype=paddle.float32)
        box, var = ops.prior_box(
            input=inp,
            image=image,
            min_sizes=[2.0, 4.0],
            clip=True,
            flip=True,
        )
        # box shape: [H, W, num_priors, 4]
        self.assertEqual(box.shape[0], 6)
        self.assertEqual(box.shape[1], 9)
        self.assertEqual(box.shape[3], 4)

    def test_prior_box_no_clip(self):
        """测试不裁剪 / Test without clipping."""
        inp = paddle.randn((1, 3, 5, 5), dtype=paddle.float32)
        image = paddle.randn((1, 3, 10, 10), dtype=paddle.float32)
        box, var = ops.prior_box(
            input=inp,
            image=image,
            min_sizes=[2.0],
            clip=False,
        )
        self.assertEqual(box.shape[0], 5)
        self.assertEqual(box.shape[1], 5)
        self.assertEqual(box.shape[3], 4)

    def test_prior_box_with_max_sizes(self):
        """测试带max_sizes / Test with max_sizes."""
        inp = paddle.randn((1, 3, 6, 9), dtype=paddle.float32)
        image = paddle.randn((1, 3, 9, 12), dtype=paddle.float32)
        box, var = ops.prior_box(
            input=inp,
            image=image,
            min_sizes=[2.0],
            max_sizes=[5.0],
            clip=False,
        )
        self.assertEqual(box.shape[0], 6)

    def test_prior_box_float_min_sizes(self):
        """测试float类型的min_sizes / Test float min_sizes."""
        inp = paddle.randn((1, 3, 4, 4), dtype=paddle.float32)
        image = paddle.randn((1, 3, 8, 8), dtype=paddle.float32)
        box, var = ops.prior_box(
            input=inp,
            image=image,
            min_sizes=3.0,
            clip=False,
        )
        self.assertEqual(box.shape[0], 4)

    def test_prior_box_min_max_aspect_ratios_order(self):
        """测试min_max_aspect_ratios_order / Test min_max_aspect_ratios_order."""
        inp = paddle.randn((1, 3, 4, 4), dtype=paddle.float32)
        image = paddle.randn((1, 3, 8, 8), dtype=paddle.float32)
        box, var = ops.prior_box(
            input=inp,
            image=image,
            min_sizes=[2.0],
            max_sizes=[5.0],
            clip=False,
            min_max_aspect_ratios_order=True,
        )
        self.assertEqual(box.shape[0], 4)


class TestBoxCoder(unittest.TestCase):
    """测试 box_coder / Test box_coder in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_box_coder_encode_with_tensor_var(self):
        """编码测试，使用Tensor类型的prior_box_var / Encode with Tensor prior_box_var."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float32)
        prior_box_var = paddle.rand((10, 4), dtype=paddle.float32)
        target_box = paddle.rand((5, 4), dtype=paddle.float32)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="encode_center_size",
            box_normalized=True,
        )
        self.assertEqual(output.shape, [5, 10, 4])

    def test_box_coder_encode_with_list_var(self):
        """编码测试，使用list类型的prior_box_var / Encode with list prior_box_var."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float32)
        prior_box_var = [0.1, 0.1, 0.2, 0.2]
        target_box = paddle.rand((5, 4), dtype=paddle.float32)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="encode_center_size",
            box_normalized=True,
        )
        self.assertEqual(output.shape, [5, 10, 4])

    def test_box_coder_decode_with_tensor_var(self):
        """解码测试，使用Tensor类型的prior_box_var / Decode with Tensor prior_box_var."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float32)
        prior_box_var = paddle.rand((10, 4), dtype=paddle.float32)
        target_box = paddle.rand((5, 10, 4), dtype=paddle.float32)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="decode_center_size",
            box_normalized=False,
            axis=0,
        )
        self.assertEqual(output.shape, [5, 10, 4])

    def test_box_coder_decode_with_list_var(self):
        """解码测试，使用list类型的prior_box_var / Decode with list prior_box_var."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float32)
        prior_box_var = [0.1, 0.1, 0.2, 0.2]
        target_box = paddle.rand((5, 10, 4), dtype=paddle.float32)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="decode_center_size",
            box_normalized=False,
        )
        self.assertEqual(output.shape, [5, 10, 4])

    def test_box_coder_decode_axis1(self):
        """测试axis=1的编码 / Test encode with axis=1."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float32)
        prior_box_var = [0.1, 0.1, 0.2, 0.2]
        # axis=1 with rank 2 inputs: both have same first dim, output is (10, 10, 4)
        target_box = paddle.rand((10, 4), dtype=paddle.float32)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="encode_center_size",
            box_normalized=True,
            axis=1,
        )
        self.assertEqual(output.shape, [10, 10, 4])

    def test_box_coder_encode_float64(self):
        """测试float64编码 / Test float64 encode."""
        prior_box = paddle.rand((10, 4), dtype=paddle.float64)
        prior_box_var = paddle.rand((10, 4), dtype=paddle.float64)
        target_box = paddle.rand((5, 4), dtype=paddle.float64)
        output = ops.box_coder(
            prior_box=prior_box,
            prior_box_var=prior_box_var,
            target_box=target_box,
            code_type="encode_center_size",
        )
        self.assertEqual(output.dtype, paddle.float64)


class TestNms(unittest.TestCase):
    """测试 nms / Test nms in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_nms_basic(self):
        """基本nms测试 / Basic nms test."""
        boxes = paddle.rand([4, 4]).astype('float32')
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        out = ops.nms(boxes, 0.1)
        self.assertEqual(out.dtype, paddle.int64)
        self.assertTrue(out.shape[0] <= 4)

    def test_nms_with_scores(self):
        """带scores的nms测试 / NMS with scores."""
        boxes = paddle.rand([4, 4]).astype('float32')
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        scores = paddle.to_tensor([0.6, 0.7, 0.4, 0.233])
        out = ops.nms(boxes, 0.1, scores=scores)
        self.assertEqual(out.dtype, paddle.int64)

    def test_nms_with_categories(self):
        """带category_idxs和categories的nms测试 / NMS with categories."""
        boxes = paddle.rand([4, 4]).astype('float32')
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        scores = paddle.to_tensor([0.6, 0.7, 0.4, 0.233])
        category_idxs = paddle.to_tensor([2, 0, 0, 3], dtype="int64")
        categories = [0, 1, 2, 3]
        out = ops.nms(
            boxes,
            0.1,
            scores=scores,
            category_idxs=category_idxs,
            categories=categories,
        )
        self.assertEqual(out.dtype, paddle.int64)

    def test_nms_with_categories_and_topk(self):
        """带categories和top_k的nms测试 / NMS with categories and top_k."""
        boxes = paddle.rand([4, 4]).astype('float32')
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        scores = paddle.to_tensor([0.6, 0.7, 0.4, 0.233])
        category_idxs = paddle.to_tensor([2, 0, 0, 3], dtype="int64")
        categories = [0, 1, 2, 3]
        out = ops.nms(
            boxes,
            0.1,
            scores=scores,
            category_idxs=category_idxs,
            categories=categories,
            top_k=4,
        )
        self.assertEqual(out.dtype, paddle.int64)
        self.assertTrue(out.shape[0] <= 4)

    def test_nms_no_overlap(self):
        """无重叠框的nms测试 / NMS with non-overlapping boxes."""
        # 创建不重叠的框 / Create non-overlapping boxes
        boxes = paddle.to_tensor(
            [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5], [6, 6, 7, 7]],
            dtype='float32',
        )
        out = ops.nms(boxes, 0.3)
        self.assertEqual(out.shape[0], 4)

    def test_nms_all_overlap(self):
        """全部重叠框的nms测试 / NMS with all overlapping boxes."""
        boxes = paddle.to_tensor(
            [[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12], [3, 3, 13, 13]],
            dtype='float32',
        )
        out = ops.nms(boxes, 0.1)
        # 重叠很大，大部分应被抑制 / High overlap => most suppressed
        self.assertTrue(out.shape[0] <= 2)


class TestMatrixNms(unittest.TestCase):
    """测试 matrix_nms / Test matrix_nms in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_matrix_nms_basic(self):
        """基本matrix_nms测试 / Basic matrix_nms test."""
        bboxes = paddle.rand([4, 1, 4], dtype='float32')
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        scores = paddle.rand([4, 80, 1], dtype='float32')
        # In dygraph mode: returns (out, rois_num, index)
        out, rois_num, index = ops.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            background_label=0,
            score_threshold=0.5,
            post_threshold=0.1,
            nms_top_k=400,
            keep_top_k=200,
            normalized=False,
            return_index=True,
            return_rois_num=True,
        )
        self.assertEqual(out.ndim, 2)
        self.assertEqual(rois_num.ndim, 1)
        self.assertEqual(index.ndim, 2)

    def test_matrix_nms_no_index(self):
        """测试不返回index / Test without returning index."""
        bboxes = paddle.rand([2, 1, 4], dtype='float32')
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        scores = paddle.rand([2, 10, 1], dtype='float32')
        out, rois_num, index = ops.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            background_label=-1,
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=100,
            keep_top_k=50,
            normalized=True,
            return_index=False,
            return_rois_num=True,
        )
        self.assertEqual(out.ndim, 2)
        self.assertIsNone(index)
        self.assertEqual(rois_num.shape[0], 2)

    def test_matrix_nms_no_rois_num(self):
        """测试不返回rois_num / Test without returning rois_num."""
        bboxes = paddle.rand([2, 1, 4], dtype='float32')
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        scores = paddle.rand([2, 5, 1], dtype='float32')
        out, rois_num, index = ops.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            background_label=-1,
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=100,
            keep_top_k=50,
            normalized=True,
            return_index=True,
            return_rois_num=False,
        )
        self.assertEqual(out.ndim, 2)
        self.assertIsNone(rois_num)
        self.assertIsNotNone(index)

    def test_matrix_nms_gaussian(self):
        """测试高斯衰减 / Test Gaussian decay."""
        bboxes = paddle.rand([2, 1, 4], dtype='float32')
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        scores = paddle.rand([2, 5, 1], dtype='float32')
        out, rois_num, index = ops.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            background_label=0,
            score_threshold=0.01,
            post_threshold=0.01,
            nms_top_k=100,
            keep_top_k=50,
            use_gaussian=True,
            gaussian_sigma=2.0,
            normalized=True,
            return_index=True,
            return_rois_num=True,
        )
        self.assertEqual(out.ndim, 2)

    def test_matrix_nms_keep_all(self):
        """测试keep_top_k=-1保留所有框 / Test keep_top_k=-1 to keep all."""
        bboxes = paddle.rand([1, 1, 4], dtype='float32')
        bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
        bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
        scores = paddle.rand([1, 3, 1], dtype='float32')
        out, rois_num, index = ops.matrix_nms(
            bboxes=bboxes,
            scores=scores,
            background_label=-1,
            score_threshold=0.0,
            post_threshold=0.0,
            nms_top_k=-1,
            keep_top_k=-1,
            normalized=True,
            return_index=True,
            return_rois_num=True,
        )
        self.assertEqual(out.ndim, 2)


class TestRoiAlign(unittest.TestCase):
    """测试 roi_align / Test roi_align in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_roi_align_basic(self):
        """基本roi_align测试 / Basic roi_align test."""
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = ops.roi_align(data, boxes, boxes_num, output_size=3)
        self.assertEqual(align_out.shape, [3, 256, 3, 3])

    def test_roi_align_tuple_output(self):
        """测试tuple output_size / Test tuple output_size."""
        data = paddle.rand([1, 128, 16, 16])
        boxes = paddle.rand([5, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([5]).astype('int32')
        align_out = ops.roi_align(data, boxes, boxes_num, output_size=(4, 3))
        self.assertEqual(align_out.shape, [5, 128, 4, 3])

    def test_roi_align_spatial_scale(self):
        """测试不同spatial_scale / Test different spatial_scale."""
        data = paddle.rand([1, 64, 20, 20])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = ops.roi_align(
            data,
            boxes,
            boxes_num,
            output_size=7,
            spatial_scale=0.5,
        )
        self.assertEqual(align_out.shape, [3, 64, 7, 7])

    def test_roi_align_sampling_ratio(self):
        """测试采样率参数 / Test sampling_ratio parameter."""
        data = paddle.rand([1, 64, 20, 20])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = ops.roi_align(
            data,
            boxes,
            boxes_num,
            output_size=7,
            sampling_ratio=2,
        )
        self.assertEqual(align_out.shape, [3, 64, 7, 7])

    def test_roi_align_not_aligned(self):
        """测试aligned=False / Test aligned=False (legacy mode)."""
        data = paddle.rand([1, 64, 20, 20])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = ops.roi_align(
            data,
            boxes,
            boxes_num,
            output_size=7,
            aligned=False,
        )
        self.assertEqual(align_out.shape, [3, 64, 7, 7])

    def test_roi_align_float64(self):
        """测试float64输入 / Test float64 input."""
        data = paddle.rand([1, 32, 10, 10], dtype='float64')
        boxes = paddle.rand([2, 4], dtype='float64')
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 3
        boxes_num = paddle.to_tensor([2]).astype('int32')
        align_out = ops.roi_align(data, boxes, boxes_num, output_size=3)
        self.assertEqual(align_out.dtype, paddle.float64)


class TestRoIAlignLayer(unittest.TestCase):
    """测试 RoIAlign 类 / Test RoIAlign as Layer."""

    def setUp(self):
        paddle.disable_static()

    def test_roi_align_layer_basic(self):
        """基本RoIAlign Layer测试 / Basic RoIAlign Layer test."""
        roi_align_layer = ops.RoIAlign(output_size=(4, 3))
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = roi_align_layer(data, boxes, boxes_num)
        self.assertEqual(align_out.shape, [3, 256, 4, 3])

    def test_roi_align_layer_int_output(self):
        """测试int类型的output_size / Test int output_size."""
        roi_align_layer = ops.RoIAlign(output_size=7)
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = roi_align_layer(data, boxes, boxes_num)
        self.assertEqual(align_out.shape, [3, 256, 7, 7])

    def test_roi_align_layer_aligned_false(self):
        """测试aligned=False / Test aligned=False."""
        roi_align_layer = ops.RoIAlign(output_size=7)
        data = paddle.rand([1, 128, 20, 20])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = roi_align_layer(data, boxes, boxes_num, aligned=False)
        self.assertEqual(align_out.shape, [3, 128, 7, 7])

    def test_roi_align_layer_spatial_scale(self):
        """测试spatial_scale参数 / Test spatial_scale."""
        roi_align_layer = ops.RoIAlign(output_size=7, spatial_scale=0.25)
        data = paddle.rand([1, 128, 40, 40])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        align_out = roi_align_layer(data, boxes, boxes_num)
        self.assertEqual(align_out.shape, [3, 128, 7, 7])


class TestRoiPool(unittest.TestCase):
    """测试 roi_pool / Test roi_pool in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_roi_pool_basic(self):
        """基本roi_pool测试 / Basic roi_pool test."""
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        pool_out = ops.roi_pool(data, boxes, boxes_num=boxes_num, output_size=3)
        self.assertEqual(pool_out.shape, [3, 256, 3, 3])

    def test_roi_pool_tuple_output(self):
        """测试tuple output_size / Test tuple output_size."""
        data = paddle.rand([1, 128, 16, 16])
        boxes = paddle.rand([5, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([5]).astype('int32')
        pool_out = ops.roi_pool(
            data, boxes, boxes_num=boxes_num, output_size=(4, 3)
        )
        self.assertEqual(pool_out.shape, [5, 128, 4, 3])

    def test_roi_pool_spatial_scale(self):
        """测试不同spatial_scale / Test different spatial_scale."""
        data = paddle.rand([1, 64, 20, 20])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        pool_out = ops.roi_pool(
            data,
            boxes,
            boxes_num=boxes_num,
            output_size=7,
            spatial_scale=0.5,
        )
        self.assertEqual(pool_out.shape, [3, 64, 7, 7])

    def test_roi_pool_multiple_batches(self):
        """测试多batch / Test multiple batches."""
        data = paddle.rand([2, 32, 16, 16])
        boxes = paddle.rand([6, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 3
        boxes_num = paddle.to_tensor([3, 3]).astype('int32')
        pool_out = ops.roi_pool(data, boxes, boxes_num=boxes_num, output_size=3)
        self.assertEqual(pool_out.shape, [6, 32, 3, 3])


class TestDeformConv2D(unittest.TestCase):
    """测试 deform_conv2d / Test deform_conv2d in dynamic graph mode."""

    def setUp(self):
        paddle.disable_static()

    def test_deform_conv2d_v1(self):
        """可变形卷积v1测试 / Deformable conv v1 test (no mask)."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        # out_h = (10 - 3) / 1 + 1 = 8
        offset = paddle.rand((2, 2 * kh * kw, 8, 8))
        out = ops.deform_conv2d(x, offset, weight)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_v2(self):
        """可变形卷积v2测试 / Deformable conv v2 test (with mask)."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        offset = paddle.rand((2, 2 * kh * kw, 8, 8))
        mask = paddle.rand((2, kh * kw, 8, 8))
        out = ops.deform_conv2d(x, offset, weight, mask=mask)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_with_bias(self):
        """测试带bias的可变形卷积 / Deform conv with bias."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        bias = paddle.zeros([16])
        offset = paddle.rand((2, 2 * kh * kw, 8, 8))
        out = ops.deform_conv2d(x, offset, weight, bias=bias)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_with_padding(self):
        """测试带padding的可变形卷积 / Deform conv with padding."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        # out_h = (10 + 2*1 - 3) / 1 + 1 = 10
        offset = paddle.rand((2, 2 * kh * kw, 10, 10))
        out = ops.deform_conv2d(x, offset, weight, stride=1, padding=1)
        self.assertEqual(out.shape, [2, 16, 10, 10])

    def test_deform_conv2d_with_stride(self):
        """测试带stride的可变形卷积 / Deform conv with stride."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        # out_h = (10 - 3) / 2 + 1 = 4 (floor)
        offset = paddle.rand((2, 2 * kh * kw, 4, 4))
        out = ops.deform_conv2d(x, offset, weight, stride=2)
        self.assertEqual(out.shape, [2, 16, 4, 4])

    def test_deform_conv2d_with_dilation(self):
        """测试带dilation的可变形卷积 / Deform conv with dilation."""
        x = paddle.rand((2, 1, 10, 10))
        kh, kw = 3, 3
        weight = paddle.rand((16, 1, kh, kw))
        # out_h = (10 - (2*(3-1)+1)) / 1 + 1 = 6
        offset = paddle.rand((2, 2 * kh * kw, 6, 6))
        out = ops.deform_conv2d(
            x, offset, weight, stride=1, padding=0, dilation=2
        )
        self.assertEqual(out.shape, [2, 16, 6, 6])

    def test_deform_conv2d_groups(self):
        """测试分组可变形卷积 / Deform conv with groups."""
        x = paddle.rand((2, 2, 10, 10))
        kh, kw = 3, 3
        # out_channels=16, in_channels=2, groups=2 => per-group: 16/2=8 out, 2/2=1 in
        weight = paddle.rand((16, 1, kh, kw))
        offset = paddle.rand((2, 2 * kh * kw, 8, 8))
        out = ops.deform_conv2d(
            x,
            offset,
            weight,
            stride=1,
            padding=0,
            groups=2,
        )
        self.assertEqual(out.shape, [2, 16, 8, 8])


class TestDeformConv2DLayer(unittest.TestCase):
    """测试 DeformConv2D 类 / Test DeformConv2D as Layer."""

    def setUp(self):
        paddle.disable_static()

    def test_deform_conv2d_layer_v1(self):
        """DeformConv2D Layer v1测试 / DeformConv2D Layer v1 test."""
        deform_conv = ops.DeformConv2D(
            in_channels=1, out_channels=16, kernel_size=[3, 3]
        )
        x = paddle.rand((2, 1, 10, 10))
        offset = paddle.rand((2, 18, 8, 8))
        out = deform_conv(x, offset)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_layer_v2(self):
        """DeformConv2D Layer v2测试 / DeformConv2D Layer v2 test."""
        deform_conv = ops.DeformConv2D(
            in_channels=1, out_channels=16, kernel_size=[3, 3]
        )
        x = paddle.rand((2, 1, 10, 10))
        offset = paddle.rand((2, 18, 8, 8))
        mask = paddle.rand((2, 9, 8, 8))
        out = deform_conv(x, offset, mask)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_layer_no_bias(self):
        """DeformConv2D Layer不带bias / DeformConv2D Layer without bias."""
        deform_conv = ops.DeformConv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=[3, 3],
            bias_attr=False,
        )
        x = paddle.rand((2, 1, 10, 10))
        offset = paddle.rand((2, 18, 8, 8))
        out = deform_conv(x, offset)
        self.assertEqual(out.shape, [2, 16, 8, 8])

    def test_deform_conv2d_layer_stride_padding(self):
        """DeformConv2D Layer带stride和padding / Layer with stride and padding."""
        deform_conv = ops.DeformConv2D(
            in_channels=1,
            out_channels=16,
            kernel_size=[3, 3],
            stride=2,
            padding=1,
        )
        x = paddle.rand((2, 1, 10, 10))
        # out_h = (10 + 2*1 - 3) / 2 + 1 = 5
        offset = paddle.rand((2, 18, 5, 5))
        out = deform_conv(x, offset)
        self.assertEqual(out.shape, [2, 16, 5, 5])


class TestRoIPoolLayer(unittest.TestCase):
    """测试 RoIPool 类 / Test RoIPool as Layer."""

    def setUp(self):
        paddle.disable_static()

    def test_roi_pool_layer_basic(self):
        """基本RoIPool Layer测试 / Basic RoIPool Layer test."""
        roi_pool_layer = ops.RoIPool(output_size=(4, 3))
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        pool_out = roi_pool_layer(data, boxes, boxes_num)
        self.assertEqual(pool_out.shape, [3, 256, 4, 3])

    def test_roi_pool_layer_int_output(self):
        """测试int类型的output_size / Test int output_size."""
        roi_pool_layer = ops.RoIPool(output_size=7)
        data = paddle.rand([1, 256, 32, 32])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 3
        boxes[:, 3] = boxes[:, 1] + 4
        boxes_num = paddle.to_tensor([3]).astype('int32')
        pool_out = roi_pool_layer(data, boxes, boxes_num)
        self.assertEqual(pool_out.shape, [3, 256, 7, 7])

    def test_roi_pool_layer_spatial_scale(self):
        """测试spatial_scale参数 / Test spatial_scale."""
        roi_pool_layer = ops.RoIPool(output_size=7, spatial_scale=0.25)
        data = paddle.rand([1, 128, 40, 40])
        boxes = paddle.rand([3, 4])
        boxes[:, 2] = boxes[:, 0] + 5
        boxes[:, 3] = boxes[:, 1] + 5
        boxes_num = paddle.to_tensor([3]).astype('int32')
        pool_out = roi_pool_layer(data, boxes, boxes_num)
        self.assertEqual(pool_out.shape, [3, 128, 7, 7])

    def test_roi_pool_layer_extra_repr(self):
        """测试extra_repr / Test extra_repr."""
        roi_pool_layer = ops.RoIPool(output_size=(4, 3), spatial_scale=0.5)
        repr_str = roi_pool_layer.extra_repr()
        self.assertIn('4', repr_str)
        self.assertIn('3', repr_str)
        self.assertIn('0.5', repr_str)


class TestDistributeFpnProposals(unittest.TestCase):
    """测试 distribute_fpn_proposals / Test distribute_fpn_proposals."""

    def setUp(self):
        paddle.disable_static()

    def test_distribute_fpn_proposals_basic(self):
        """基本测试 / Basic test."""
        fpn_rois = paddle.rand((10, 4), dtype=paddle.float32)
        rois_num = paddle.to_tensor([3, 1, 4, 2], dtype=paddle.int32)
        multi_rois, restore_ind, rois_num_per_level = (
            ops.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num,
            )
        )
        # 应返回4个level (max_level - min_level + 1 = 4)
        self.assertEqual(len(multi_rois), 4)
        self.assertEqual(restore_ind.shape[1], 1)


class TestGenerateProposals(unittest.TestCase):
    """测试 generate_proposals / Test generate_proposals."""

    def setUp(self):
        paddle.disable_static()

    def test_generate_proposals_basic(self):
        """基本测试 / Basic test."""
        scores = paddle.rand((2, 4, 5, 5), dtype=paddle.float32)
        bbox_deltas = paddle.rand((2, 16, 5, 5), dtype=paddle.float32)
        img_size = paddle.to_tensor([[224.0, 224.0], [224.0, 224.0]])
        anchors = paddle.rand((5, 5, 4, 4), dtype=paddle.float32)
        variances = paddle.rand((5, 5, 4, 4), dtype=paddle.float32)
        rois, roi_probs, roi_nums = ops.generate_proposals(
            scores,
            bbox_deltas,
            img_size,
            anchors,
            variances,
            return_rois_num=True,
        )
        self.assertEqual(rois.ndim, 2)
        self.assertEqual(roi_probs.ndim, 2)
        self.assertEqual(roi_nums.ndim, 1)

    def test_generate_proposals_with_pixel_offset(self):
        """测试pixel_offset / Test with pixel_offset."""
        scores = paddle.rand((1, 4, 5, 5), dtype=paddle.float32)
        bbox_deltas = paddle.rand((1, 16, 5, 5), dtype=paddle.float32)
        img_size = paddle.to_tensor([[224.0, 224.0]])
        anchors = paddle.rand((5, 5, 4, 4), dtype=paddle.float32)
        variances = paddle.rand((5, 5, 4, 4), dtype=paddle.float32)
        rois, roi_probs, roi_nums = ops.generate_proposals(
            scores,
            bbox_deltas,
            img_size,
            anchors,
            variances,
            pixel_offset=True,
            return_rois_num=True,
        )
        self.assertEqual(rois.ndim, 2)


class TestPsRoiPool(unittest.TestCase):
    """测试 psroi_pool / Test psroi_pool."""

    def setUp(self):
        paddle.disable_static()

    def test_psroi_pool_basic(self):
        """基本测试 / Basic test."""
        x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]],
            dtype='float32',
        )
        boxes_num = paddle.to_tensor([1, 2], dtype='int32')
        pool_out = ops.psroi_pool(x, boxes, boxes_num, 7, 1.0)
        self.assertEqual(pool_out.shape, [3, 10, 7, 7])

    def test_psroi_pool_int_output(self):
        """测试int output_size / Test int output_size."""
        x = paddle.uniform([2, 100, 14, 14], dtype='float32')
        boxes = paddle.to_tensor(
            [[1, 1, 5, 5], [2, 2, 6, 6], [8, 8, 12, 12]],
            dtype='float32',
        )
        boxes_num = paddle.to_tensor([1, 2], dtype='int32')
        pool_out = ops.psroi_pool(x, boxes, boxes_num, output_size=5)
        self.assertEqual(pool_out.shape, [3, 4, 5, 5])


class TestPSRoIPoolLayer(unittest.TestCase):
    """测试 PSRoIPool 类 / Test PSRoIPool as Layer."""

    def setUp(self):
        paddle.disable_static()

    def test_psroi_pool_layer_basic(self):
        """基本PSRoIPool Layer测试 / Basic PSRoIPool Layer test."""
        psroi_module = ops.PSRoIPool(7, 1.0)
        x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]],
            dtype='float32',
        )
        boxes_num = paddle.to_tensor([1, 2], dtype='int32')
        pool_out = psroi_module(x, boxes, boxes_num)
        self.assertEqual(pool_out.shape, [3, 10, 7, 7])

    def test_psroi_pool_layer_spatial_scale(self):
        """测试spatial_scale / Test spatial_scale."""
        psroi_module = ops.PSRoIPool(7, 0.5)
        x = paddle.uniform([2, 490, 28, 28], dtype='float32')
        boxes = paddle.to_tensor(
            [[1, 5, 8, 10], [4, 2, 6, 7], [12, 12, 19, 21]],
            dtype='float32',
        )
        boxes_num = paddle.to_tensor([1, 2], dtype='int32')
        pool_out = psroi_module(x, boxes, boxes_num)
        self.assertEqual(pool_out.shape, [3, 10, 7, 7])


if __name__ == '__main__':
    unittest.main()
