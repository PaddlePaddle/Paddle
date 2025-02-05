#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

import numpy as np
from op_test import OpTest
from scipy.special import expit, logit

import paddle
from paddle.base import core


def l1loss(x, y):
    return abs(x - y)


def sce(x, label):
    sigmoid_x = expit(x)
    term1 = label * np.log(sigmoid_x)
    term2 = (1.0 - label) * np.log(1.0 - sigmoid_x)
    return -term1 - term2


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def batch_xywh_box_iou(box1, box2):
    b1_left = box1[:, :, 0] - box1[:, :, 2] / 2
    b1_right = box1[:, :, 0] + box1[:, :, 2] / 2
    b1_top = box1[:, :, 1] - box1[:, :, 3] / 2
    b1_bottom = box1[:, :, 1] + box1[:, :, 3] / 2

    b2_left = box2[:, :, 0] - box2[:, :, 2] / 2
    b2_right = box2[:, :, 0] + box2[:, :, 2] / 2
    b2_top = box2[:, :, 1] - box2[:, :, 3] / 2
    b2_bottom = box2[:, :, 1] + box2[:, :, 3] / 2

    left = np.maximum(b1_left[:, :, np.newaxis], b2_left[:, np.newaxis, :])
    right = np.minimum(b1_right[:, :, np.newaxis], b2_right[:, np.newaxis, :])
    top = np.maximum(b1_top[:, :, np.newaxis], b2_top[:, np.newaxis, :])
    bottom = np.minimum(
        b1_bottom[:, :, np.newaxis], b2_bottom[:, np.newaxis, :]
    )

    inter_w = np.clip(right - left, 0.0, 1.0)
    inter_h = np.clip(bottom - top, 0.0, 1.0)
    inter_area = inter_w * inter_h

    b1_area = (b1_right - b1_left) * (b1_bottom - b1_top)
    b2_area = (b2_right - b2_left) * (b2_bottom - b2_top)
    union = b1_area[:, :, np.newaxis] + b2_area[:, np.newaxis, :] - inter_area

    return inter_area / union


def YOLOv3Loss(x, gtbox, gtlabel, gtscore, attrs):
    n, c, h, w = x.shape
    b = gtbox.shape[1]
    anchors = attrs['anchors']
    an_num = len(anchors) // 2
    anchor_mask = attrs['anchor_mask']
    mask_num = len(anchor_mask)
    class_num = attrs["class_num"]
    ignore_thresh = attrs['ignore_thresh']
    downsample_ratio = attrs['downsample_ratio']
    use_label_smooth = attrs['use_label_smooth']
    scale_x_y = attrs['scale_x_y']
    bias_x_y = -0.5 * (scale_x_y - 1.0)
    input_size = downsample_ratio * h
    x = x.reshape((n, mask_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))
    loss = np.zeros(n).astype('float64')

    smooth_weight = min(1.0 / class_num, 1.0 / 40)
    label_pos = 1.0 - smooth_weight if use_label_smooth else 1.0
    label_neg = smooth_weight if use_label_smooth else 0.0

    pred_box = x[:, :, :, :, :4].copy()
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    pred_box[:, :, :, :, 0] = (
        grid_x + sigmoid(pred_box[:, :, :, :, 0]) * scale_x_y + bias_x_y
    ) / w
    pred_box[:, :, :, :, 1] = (
        grid_y + sigmoid(pred_box[:, :, :, :, 1]) * scale_x_y + bias_x_y
    ) / h

    mask_anchors = []
    for m in anchor_mask:
        mask_anchors.append((anchors[2 * m], anchors[2 * m + 1]))
    anchors_s = np.array(
        [(an_w / input_size, an_h / input_size) for an_w, an_h in mask_anchors]
    )
    anchor_w = anchors_s[:, 0:1].reshape((1, mask_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, mask_num, 1, 1))
    pred_box[:, :, :, :, 2] = np.exp(pred_box[:, :, :, :, 2]) * anchor_w
    pred_box[:, :, :, :, 3] = np.exp(pred_box[:, :, :, :, 3]) * anchor_h

    pred_box = pred_box.reshape((n, -1, 4))
    pred_obj = x[:, :, :, :, 4].reshape((n, -1))
    objness = np.zeros(pred_box.shape[:2]).astype('float64')
    ious = batch_xywh_box_iou(pred_box, gtbox)
    ious_max = np.max(ious, axis=-1)
    objness = np.where(
        ious_max > ignore_thresh, -np.ones_like(objness), objness
    )

    gtbox_shift = gtbox.copy()
    gtbox_shift[:, :, 0] = 0
    gtbox_shift[:, :, 1] = 0

    anchors = [(anchors[2 * i], anchors[2 * i + 1]) for i in range(0, an_num)]
    anchors_s = np.array(
        [(an_w / input_size, an_h / input_size) for an_w, an_h in anchors]
    )
    anchor_boxes = np.concatenate(
        [np.zeros_like(anchors_s), anchors_s], axis=-1
    )
    anchor_boxes = np.tile(anchor_boxes[np.newaxis, :, :], (n, 1, 1))
    ious = batch_xywh_box_iou(gtbox_shift, anchor_boxes)
    iou_matches = np.argmax(ious, axis=-1)
    gt_matches = iou_matches.copy()
    for i in range(n):
        for j in range(b):
            if gtbox[i, j, 2:].sum() == 0:
                gt_matches[i, j] = -1
                continue
            if iou_matches[i, j] not in anchor_mask:
                gt_matches[i, j] = -1
                continue
            an_idx = anchor_mask.index(iou_matches[i, j])
            gt_matches[i, j] = an_idx
            gi = int(gtbox[i, j, 0] * w)
            gj = int(gtbox[i, j, 1] * h)

            tx = gtbox[i, j, 0] * w - gi
            ty = gtbox[i, j, 1] * w - gj
            tw = np.log(gtbox[i, j, 2] * input_size / mask_anchors[an_idx][0])
            th = np.log(gtbox[i, j, 3] * input_size / mask_anchors[an_idx][1])
            scale = (2.0 - gtbox[i, j, 2] * gtbox[i, j, 3]) * gtscore[i, j]
            loss[i] += sce(x[i, an_idx, gj, gi, 0], tx) * scale
            loss[i] += sce(x[i, an_idx, gj, gi, 1], ty) * scale
            loss[i] += l1loss(x[i, an_idx, gj, gi, 2], tw) * scale
            loss[i] += l1loss(x[i, an_idx, gj, gi, 3], th) * scale

            objness[i, an_idx * h * w + gj * w + gi] = gtscore[i, j]

            for label_idx in range(class_num):
                loss[i] += (
                    sce(
                        x[i, an_idx, gj, gi, 5 + label_idx],
                        label_pos if label_idx == gtlabel[i, j] else label_neg,
                    )
                    * gtscore[i, j]
                )

        for j in range(mask_num * h * w):
            if objness[i, j] > 0:
                loss[i] += sce(pred_obj[i, j], 1.0) * objness[i, j]
            elif objness[i, j] == 0:
                loss[i] += sce(pred_obj[i, j], 0.0)

    return (
        loss,
        objness.reshape((n, mask_num, h, w)).astype('float64'),
        gt_matches.astype('int32'),
    )


def yolo_loss_wrapper(
    x,
    gt_box,
    gt_label,
    gt_score=None,
    anchors=[
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
    ],
    anchor_mask=[0, 1, 2],
    class_num=5,
    ignore_thresh=0.7,
    downsample_ratio=32,
    use_label_smooth=True,
    scale_x_y=1.0,
):
    loss = paddle.vision.ops.yolo_loss(
        x,
        gt_box=gt_box,
        gt_label=gt_label,
        anchors=anchors,
        anchor_mask=anchor_mask,
        class_num=class_num,
        ignore_thresh=ignore_thresh,
        downsample_ratio=downsample_ratio,
        gt_score=gt_score,
        use_label_smooth=use_label_smooth,
        scale_x_y=scale_x_y,
    )
    return loss


class TestYolov3LossOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'yolov3_loss'
        self.python_api = yolo_loss_wrapper
        self.python_out_sig = ['Loss']
        x = logit(np.random.uniform(0, 1, self.x_shape).astype('float64'))
        gtbox = np.random.random(size=self.gtbox_shape).astype('float64')
        gtlabel = np.random.randint(0, self.class_num, self.gtbox_shape[:2])
        gtmask = np.random.randint(0, 2, self.gtbox_shape[:2])
        gtbox = gtbox * gtmask[:, :, np.newaxis]
        gtlabel = gtlabel * gtmask

        self.attrs = {
            "anchors": self.anchors,
            "anchor_mask": self.anchor_mask,
            "class_num": self.class_num,
            "ignore_thresh": self.ignore_thresh,
            "downsample_ratio": self.downsample_ratio,
            "use_label_smooth": self.use_label_smooth,
            "scale_x_y": self.scale_x_y,
        }

        self.inputs = {
            'X': x,
            'GTBox': gtbox.astype('float64'),
            'GTLabel': gtlabel.astype('int32'),
        }

        gtscore = np.ones(self.gtbox_shape[:2]).astype('float64')
        if self.gtscore:
            gtscore = np.random.random(self.gtbox_shape[:2]).astype('float64')
            self.inputs['GTScore'] = gtscore

        loss, objness, gt_matches = YOLOv3Loss(
            x, gtbox, gtlabel, gtscore, self.attrs
        )
        self.outputs = {
            'Loss': loss,
            'ObjectnessMask': objness,
            "GTMatchMask": gt_matches,
        }

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=2e-3, check_pir=True)

    def test_check_grad_ignore_gtbox(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['X'], 'Loss', max_relative_error=0.2, check_pir=True
        )

    def initTestCase(self):
        self.anchors = [
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
        self.anchor_mask = [0, 1, 2]
        self.class_num = 5
        self.ignore_thresh = 0.7
        self.downsample_ratio = 32
        self.x_shape = (3, len(self.anchor_mask) * (5 + self.class_num), 5, 5)
        self.gtbox_shape = (3, 5, 4)
        self.gtscore = True
        self.use_label_smooth = True
        self.scale_x_y = 1.0


class TestYolov3LossWithoutLabelSmooth(TestYolov3LossOp):
    def initTestCase(self):
        self.anchors = [
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
        self.anchor_mask = [0, 1, 2]
        self.class_num = 5
        self.ignore_thresh = 0.7
        self.downsample_ratio = 32
        self.x_shape = (3, len(self.anchor_mask) * (5 + self.class_num), 5, 5)
        self.gtbox_shape = (3, 5, 4)
        self.gtscore = True
        self.use_label_smooth = False
        self.scale_x_y = 1.0


class TestYolov3LossNoGTScore(TestYolov3LossOp):
    def initTestCase(self):
        self.anchors = [
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
        self.anchor_mask = [0, 1, 2]
        self.class_num = 5
        self.ignore_thresh = 0.7
        self.downsample_ratio = 32
        self.x_shape = (3, len(self.anchor_mask) * (5 + self.class_num), 5, 5)
        self.gtbox_shape = (3, 5, 4)
        self.gtscore = False
        self.use_label_smooth = True
        self.scale_x_y = 1.0


class TestYolov3LossWithScaleXY(TestYolov3LossOp):
    def initTestCase(self):
        self.anchors = [
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
        self.anchor_mask = [0, 1, 2]
        self.class_num = 5
        self.ignore_thresh = 0.7
        self.downsample_ratio = 32
        self.x_shape = (3, len(self.anchor_mask) * (5 + self.class_num), 5, 5)
        self.gtbox_shape = (3, 5, 4)
        self.gtscore = True
        self.use_label_smooth = True
        self.scale_x_y = 1.2


class TestYolov3LossDygraph(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = np.random.random([2, 14, 8, 8]).astype('float32')
        gt_box = np.random.random([2, 10, 4]).astype('float32')
        gt_label = np.random.random([2, 10]).astype('int32')

        x = paddle.to_tensor(x)
        gt_box = paddle.to_tensor(gt_box)
        gt_label = paddle.to_tensor(gt_label)

        loss = paddle.vision.ops.yolo_loss(
            x,
            gt_box=gt_box,
            gt_label=gt_label,
            anchors=[10, 13, 16, 30],
            anchor_mask=[0, 1],
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=8,
            use_label_smooth=True,
            scale_x_y=1.0,
        )
        assert loss is not None
        assert loss.shape == [2]
        paddle.enable_static()


class TestYolov3LossStatic(unittest.TestCase):

    def test_static(self):
        x = paddle.static.data('x', [2, 14, 8, 8], 'float32')
        gt_box = paddle.static.data('gt_box', [2, 10, 4], 'float32')
        gt_label = paddle.static.data('gt_label', [2, 10], 'int32')
        gt_score = paddle.static.data('gt_score', [2, 10], 'float32')

        loss = paddle.vision.ops.yolo_loss(
            x,
            gt_box=gt_box,
            gt_label=gt_label,
            anchors=[10, 13, 16, 30],
            anchor_mask=[0, 1],
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=8,
            gt_score=gt_score,
            use_label_smooth=True,
            scale_x_y=1.0,
        )
        assert loss is not None

        loss = paddle.vision.ops.yolo_loss(
            x,
            gt_box=gt_box,
            gt_label=gt_label,
            anchors=[10, 13, 16, 30],
            anchor_mask=[0, 1],
            class_num=2,
            ignore_thresh=0.7,
            downsample_ratio=8,
            use_label_smooth=True,
            scale_x_y=1.0,
        )
        assert loss is not None


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
