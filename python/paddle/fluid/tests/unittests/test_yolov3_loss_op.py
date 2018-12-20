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

from __future__ import division

import unittest
import numpy as np
from scipy.special import logit
from scipy.special import expit
from op_test import OpTest

from paddle.fluid import core

# def l1loss(x, y, weight):
#     n = x.shape[0]
#     x = x.reshape((n, -1))
#     y = y.reshape((n, -1))
#     weight = weight.reshape((n, -1))
#     return (np.abs(y - x) * weight).sum(axis=1)
#
#
# def mse(x, y, weight):
#     n = x.shape[0]
#     x = x.reshape((n, -1))
#     y = y.reshape((n, -1))
#     weight = weight.reshape((n, -1))
#     return ((y - x)**2 * weight).sum(axis=1)
#
#
# def sce(x, label, weight):
#     n = x.shape[0]
#     x = x.reshape((n, -1))
#     label = label.reshape((n, -1))
#     weight = weight.reshape((n, -1))
#     sigmoid_x = expit(x)
#     term1 = label * np.log(sigmoid_x)
#     term2 = (1.0 - label) * np.log(1.0 - sigmoid_x)
#     return ((-term1 - term2) * weight).sum(axis=1)


def l1loss(x, y):
    return abs(x - y)


def sce(x, label):
    sigmoid_x = expit(x)
    term1 = label * np.log(sigmoid_x)
    term2 = (1.0 - label) * np.log(1.0 - sigmoid_x)
    return -term1 - term2


def box_iou(box1, box2):
    b1_x1 = box1[0] - box1[2] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_y2 = box1[1] + box1[3] / 2
    b2_x1 = box2[0] - box2[2] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_y2 = box2[1] + box2[3] / 2

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    inter_area = max(inter_rect_x2 - inter_rect_x1, 0) * max(
        inter_rect_y2 - inter_rect_y1, 0)

    return inter_area / (b1_area + b2_area + inter_area)


def build_target(gtboxes, gtlabel, attrs, grid_size):
    n, b, _ = gtboxes.shape
    ignore_thresh = attrs["ignore_thresh"]
    anchors = attrs["anchors"]
    class_num = attrs["class_num"]
    input_size = attrs["input_size"]
    an_num = len(anchors) // 2
    conf_mask = np.ones((n, an_num, grid_size, grid_size)).astype('float32')
    obj_mask = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tx = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    ty = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tw = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    th = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tweight = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tconf = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tcls = np.zeros(
        (n, an_num, grid_size, grid_size, class_num)).astype('float32')

    for i in range(n):
        for j in range(b):
            if gtboxes[i, j, :].sum() == 0:
                continue

            gt_label = gtlabel[i, j]
            gx = gtboxes[i, j, 0] * grid_size
            gy = gtboxes[i, j, 1] * grid_size
            gw = gtboxes[i, j, 2] * input_size
            gh = gtboxes[i, j, 3] * input_size

            gi = int(gx)
            gj = int(gy)

            gtbox = [0, 0, gw, gh]
            max_iou = 0
            for k in range(an_num):
                anchor_box = [0, 0, anchors[2 * k], anchors[2 * k + 1]]
                iou = box_iou(gtbox, anchor_box)
                if iou > max_iou:
                    max_iou = iou
                    best_an_index = k
                if iou > ignore_thresh:
                    conf_mask[i, best_an_index, gj, gi] = 0

            conf_mask[i, best_an_index, gj, gi] = 1
            obj_mask[i, best_an_index, gj, gi] = 1
            tx[i, best_an_index, gj, gi] = gx - gi
            ty[i, best_an_index, gj, gi] = gy - gj
            tw[i, best_an_index, gj, gi] = np.log(gw / anchors[2 *
                                                               best_an_index])
            th[i, best_an_index, gj, gi] = np.log(
                gh / anchors[2 * best_an_index + 1])
            tweight[i, best_an_index, gj, gi] = 2.0 - gtboxes[
                i, j, 2] * gtboxes[i, j, 3]
            tconf[i, best_an_index, gj, gi] = 1
            tcls[i, best_an_index, gj, gi, gt_label] = 1

    return (tx, ty, tw, th, tweight, tconf, tcls, conf_mask, obj_mask)


def YoloV3Loss(x, gtbox, gtlabel, attrs):
    n, c, h, w = x.shape
    an_num = len(attrs['anchors']) // 2
    class_num = attrs["class_num"]
    x = x.reshape((n, an_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))
    pred_x = x[:, :, :, :, 0]
    pred_y = x[:, :, :, :, 1]
    pred_w = x[:, :, :, :, 2]
    pred_h = x[:, :, :, :, 3]
    pred_conf = x[:, :, :, :, 4]
    pred_cls = x[:, :, :, :, 5:]

    tx, ty, tw, th, tweight, tconf, tcls, conf_mask, obj_mask = build_target(
        gtbox, gtlabel, attrs, x.shape[2])

    obj_weight = obj_mask * tweight
    obj_mask_expand = np.tile(
        np.expand_dims(obj_mask, 4), (1, 1, 1, 1, int(attrs['class_num'])))
    loss_x = sce(pred_x, tx, obj_weight)
    loss_y = sce(pred_y, ty, obj_weight)
    loss_w = l1loss(pred_w, tw, obj_weight)
    loss_h = l1loss(pred_h, th, obj_weight)
    loss_obj = sce(pred_conf, tconf, conf_mask)
    loss_class = sce(pred_cls, tcls, obj_mask_expand)

    return loss_x + loss_y + loss_w + loss_h + loss_obj + loss_class


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
    bottom = np.minimum(b1_bottom[:, :, np.newaxis],
                        b2_bottom[:, np.newaxis, :])

    inter_w = np.clip(right - left, 0., 1.)
    inter_h = np.clip(bottom - top, 0., 1.)
    inter_area = inter_w * inter_h

    b1_area = (b1_right - b1_left) * (b1_bottom - b1_top)
    b2_area = (b2_right - b2_left) * (b2_bottom - b2_top)
    union = b1_area[:, :, np.newaxis] + b2_area[:, np.newaxis, :] - inter_area

    return inter_area / union


def YOLOv3Loss(x, gtbox, gtlabel, attrs):
    n, c, h, w = x.shape
    b = gtbox.shape[1]
    anchors = attrs['anchors']
    an_num = len(anchors) // 2
    anchor_mask = attrs['anchor_mask']
    mask_num = len(anchor_mask)
    class_num = attrs["class_num"]
    ignore_thresh = attrs['ignore_thresh']
    downsample = attrs['downsample']
    input_size = downsample * h
    x = x.reshape((n, mask_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))
    loss = np.zeros((n)).astype('float32')

    pred_box = x[:, :, :, :, :4].copy()
    grid_x = np.tile(np.arange(w).reshape((1, w)), (h, 1))
    grid_y = np.tile(np.arange(h).reshape((h, 1)), (1, w))
    pred_box[:, :, :, :, 0] = (grid_x + sigmoid(pred_box[:, :, :, :, 0])) / w
    pred_box[:, :, :, :, 1] = (grid_y + sigmoid(pred_box[:, :, :, :, 1])) / h

    mask_anchors = []
    for m in anchor_mask:
        mask_anchors.append((anchors[2 * m], anchors[2 * m + 1]))
    anchors_s = np.array(
        [(an_w / input_size, an_h / input_size) for an_w, an_h in mask_anchors])
    anchor_w = anchors_s[:, 0:1].reshape((1, mask_num, 1, 1))
    anchor_h = anchors_s[:, 1:2].reshape((1, mask_num, 1, 1))
    pred_box[:, :, :, :, 2] = np.exp(pred_box[:, :, :, :, 2]) * anchor_w
    pred_box[:, :, :, :, 3] = np.exp(pred_box[:, :, :, :, 3]) * anchor_h

    pred_box = pred_box.reshape((n, -1, 4))
    pred_obj = x[:, :, :, :, 4].reshape((n, -1))
    objness = np.zeros(pred_box.shape[:2])
    ious = batch_xywh_box_iou(pred_box, gtbox)
    ious_max = np.max(ious, axis=-1)
    objness = np.where(ious_max > ignore_thresh, -np.ones_like(objness),
                       objness)

    gtbox_shift = gtbox.copy()
    gtbox_shift[:, :, 0] = 0
    gtbox_shift[:, :, 1] = 0

    anchors = [(anchors[2 * i], anchors[2 * i + 1]) for i in range(0, an_num)]
    anchors_s = np.array(
        [(an_w / input_size, an_h / input_size) for an_w, an_h in anchors])
    anchor_boxes = np.concatenate(
        [np.zeros_like(anchors_s), anchors_s], axis=-1)
    anchor_boxes = np.tile(anchor_boxes[np.newaxis, :, :], (n, 1, 1))
    ious = batch_xywh_box_iou(gtbox_shift, anchor_boxes)
    iou_matches = np.argmax(ious, axis=-1)
    for i in range(n):
        for j in range(b):
            if gtbox[i, j, 2:].sum() == 0:
                continue
            if iou_matches[i, j] not in anchor_mask:
                continue
            an_idx = anchor_mask.index(iou_matches[i, j])
            gi = int(gtbox[i, j, 0] * w)
            gj = int(gtbox[i, j, 1] * h)

            tx = gtbox[i, j, 0] * w - gi
            ty = gtbox[i, j, 1] * w - gj
            tw = np.log(gtbox[i, j, 2] * input_size / mask_anchors[an_idx][0])
            th = np.log(gtbox[i, j, 3] * input_size / mask_anchors[an_idx][1])
            scale = 2.0 - gtbox[i, j, 2] * gtbox[i, j, 3]
            loss[i] += sce(x[i, an_idx, gj, gi, 0], tx) * scale
            loss[i] += sce(x[i, an_idx, gj, gi, 1], ty) * scale
            loss[i] += l1loss(x[i, an_idx, gj, gi, 2], tw) * scale
            loss[i] += l1loss(x[i, an_idx, gj, gi, 3], th) * scale

            objness[i, an_idx * h * w + gj * w + gi] = 1

            for label_idx in range(class_num):
                loss[i] += sce(x[i, an_idx, gj, gi, 5 + label_idx],
                               int(label_idx == gtlabel[i, j]))

        for j in range(mask_num * h * w):
            if objness[i, j] >= 0:
                loss[i] += sce(pred_obj[i, j], objness[i, j])

    return loss


class TestYolov3LossOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'yolov3_loss'
        x = logit(np.random.uniform(0, 1, self.x_shape).astype('float32'))
        gtbox = np.random.random(size=self.gtbox_shape).astype('float32')
        gtlabel = np.random.randint(0, self.class_num,
                                    self.gtbox_shape[:2]).astype('int32')

        self.attrs = {
            "anchors": self.anchors,
            "anchor_mask": self.anchor_mask,
            "class_num": self.class_num,
            "ignore_thresh": self.ignore_thresh,
            "downsample": self.downsample,
        }

        self.inputs = {'X': x, 'GTBox': gtbox, 'GTLabel': gtlabel}
        self.outputs = {'Loss': YOLOv3Loss(x, gtbox, gtlabel, self.attrs)}

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    def test_check_grad_ignore_gtbox(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['X'],
            'Loss',
            no_grad_set=set(["GTBox", "GTLabel"]),
            max_relative_error=0.15)

    def initTestCase(self):
        self.anchors = [
            10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198,
            373, 326
        ]
        self.anchor_mask = [0, 1, 2]
        self.class_num = 5
        self.ignore_thresh = 0.7
        self.downsample = 32
        self.x_shape = (3, len(self.anchor_mask) * (5 + self.class_num), 5, 5)
        self.gtbox_shape = (3, 10, 4)


if __name__ == "__main__":
    unittest.main()
