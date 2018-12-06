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
from op_test import OpTest

from paddle.fluid import core


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-1.0 * x))


def mse(x, y, num):
    return ((y - x)**2).sum() / num


def bce(x, y, mask):
    x = x.reshape((-1))
    y = y.reshape((-1))
    mask = mask.reshape((-1))

    error_sum = 0.0
    count = 0
    for i in range(x.shape[0]):
        if mask[i] > 0:
            error_sum += y[i] * np.log(x[i]) + (1 - y[i]) * np.log(1 - x[i])
            count += 1
    return error_sum / (-1.0 * count)


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


def build_target(gtboxs, gtlabel, attrs, grid_size):
    n, b, _ = gtboxs.shape
    ignore_thresh = attrs["ignore_thresh"]
    anchors = attrs["anchors"]
    class_num = attrs["class_num"]
    an_num = len(anchors) // 2
    obj_mask = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    noobj_mask = np.ones((n, an_num, grid_size, grid_size)).astype('float32')
    tx = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    ty = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tw = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    th = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tconf = np.zeros((n, an_num, grid_size, grid_size)).astype('float32')
    tcls = np.zeros(
        (n, an_num, grid_size, grid_size, class_num)).astype('float32')

    for i in range(n):
        for j in range(b):
            if gtboxs[i, j, :].sum() == 0:
                continue

            gt_label = gtlabel[i, j]
            gx = gtboxs[i, j, 0] * grid_size
            gy = gtboxs[i, j, 1] * grid_size
            gw = gtboxs[i, j, 2] * grid_size
            gh = gtboxs[i, j, 3] * grid_size

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
                    noobj_mask[i, best_an_index, gj, gi] = 0

            obj_mask[i, best_an_index, gj, gi] = 1
            noobj_mask[i, best_an_index, gj, gi] = 0
            tx[i, best_an_index, gj, gi] = gx - gi
            ty[i, best_an_index, gj, gi] = gy - gj
            tw[i, best_an_index, gj, gi] = np.log(gw / anchors[2 *
                                                               best_an_index])
            th[i, best_an_index, gj, gi] = np.log(
                gh / anchors[2 * best_an_index + 1])
            tconf[i, best_an_index, gj, gi] = 1
            tcls[i, best_an_index, gj, gi, gt_label] = 1

    return (tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask)


def YoloV3Loss(x, gtbox, gtlabel, attrs):
    n, c, h, w = x.shape
    an_num = len(attrs['anchors']) // 2
    class_num = attrs["class_num"]
    x = x.reshape((n, an_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))
    pred_x = sigmoid(x[:, :, :, :, 0])
    pred_y = sigmoid(x[:, :, :, :, 1])
    pred_w = x[:, :, :, :, 2]
    pred_h = x[:, :, :, :, 3]
    pred_conf = sigmoid(x[:, :, :, :, 4])
    pred_cls = sigmoid(x[:, :, :, :, 5:])

    tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask = build_target(
        gtbox, gtlabel, attrs, x.shape[2])

    obj_mask_expand = np.tile(
        np.expand_dims(obj_mask, 4), (1, 1, 1, 1, int(attrs['class_num'])))
    loss_x = mse(pred_x * obj_mask, tx * obj_mask, obj_mask.sum())
    loss_y = mse(pred_y * obj_mask, ty * obj_mask, obj_mask.sum())
    loss_w = mse(pred_w * obj_mask, tw * obj_mask, obj_mask.sum())
    loss_h = mse(pred_h * obj_mask, th * obj_mask, obj_mask.sum())
    loss_conf_target = bce(pred_conf * obj_mask, tconf * obj_mask, obj_mask)
    loss_conf_notarget = bce(pred_conf * noobj_mask, tconf * noobj_mask,
                             noobj_mask)
    loss_class = bce(pred_cls * obj_mask_expand, tcls * obj_mask_expand,
                     obj_mask_expand)

    return attrs['loss_weight_xy'] * (loss_x + loss_y) \
            + attrs['loss_weight_wh'] * (loss_w + loss_h) \
            + attrs['loss_weight_conf_target'] * loss_conf_target \
            + attrs['loss_weight_conf_notarget'] * loss_conf_notarget \
            + attrs['loss_weight_class'] * loss_class


class TestYolov3LossOp(OpTest):
    def setUp(self):
        self.loss_weight_xy = 1.0
        self.loss_weight_wh = 1.0
        self.loss_weight_conf_target = 1.0
        self.loss_weight_conf_notarget = 1.0
        self.loss_weight_class = 1.0
        self.initTestCase()
        self.op_type = 'yolov3_loss'
        x = np.random.random(size=self.x_shape).astype('float32')
        gtbox = np.random.random(size=self.gtbox_shape).astype('float32')
        gtlabel = np.random.randint(0, self.class_num,
                                    self.gtbox_shape[:2]).astype('int32')

        self.attrs = {
            "anchors": self.anchors,
            "class_num": self.class_num,
            "ignore_thresh": self.ignore_thresh,
            "loss_weight_xy": self.loss_weight_xy,
            "loss_weight_wh": self.loss_weight_wh,
            "loss_weight_conf_target": self.loss_weight_conf_target,
            "loss_weight_conf_notarget": self.loss_weight_conf_notarget,
            "loss_weight_class": self.loss_weight_class,
        }

        self.inputs = {'X': x, 'GTBox': gtbox, 'GTLabel': gtlabel}
        self.outputs = {
            'Loss': np.array(
                [YoloV3Loss(x, gtbox, gtlabel, self.attrs)]).astype('float32')
        }

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    def test_check_grad_ignore_gtbox(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['X'],
            'Loss',
            no_grad_set=set(["GTBox", "GTLabel"]),
            max_relative_error=0.06)

    def initTestCase(self):
        self.anchors = [10, 13, 12, 12]
        self.class_num = 10
        self.ignore_thresh = 0.5
        self.x_shape = (5, len(self.anchors) // 2 * (5 + self.class_num), 7, 7)
        self.gtbox_shape = (5, 10, 4)
        self.loss_weight_xy = 2.5
        self.loss_weight_wh = 0.8
        self.loss_weight_conf_target = 1.5
        self.loss_weight_conf_notarget = 0.5
        self.loss_weight_class = 1.2


if __name__ == "__main__":
    unittest.main()
