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


def build_target(gtboxs, attrs, grid_size):
    n, b, _ = gtboxs.shape
    ignore_thresh = attrs["ignore_thresh"]
    img_height = attrs["img_height"]
    anchors = attrs["anchors"]
    class_num = attrs["class_num"]
    an_num = len(anchors) / 2
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

            gt_label = int(gtboxs[i, j, 0])
            gx = gtboxs[i, j, 1] * grid_size
            gy = gtboxs[i, j, 2] * grid_size
            gw = gtboxs[i, j, 3] * grid_size
            gh = gtboxs[i, j, 4] * grid_size

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


def YoloV3Loss(x, gtbox, attrs):
    n, c, h, w = x.shape
    an_num = len(attrs['anchors']) / 2
    class_num = attrs["class_num"]
    x = x.reshape((n, an_num, 5 + class_num, h, w)).transpose((0, 1, 3, 4, 2))
    pred_x = sigmoid(x[:, :, :, :, 0])
    pred_y = sigmoid(x[:, :, :, :, 1])
    pred_w = x[:, :, :, :, 2]
    pred_h = x[:, :, :, :, 3]
    pred_conf = sigmoid(x[:, :, :, :, 4])
    pred_cls = sigmoid(x[:, :, :, :, 5:])

    tx, ty, tw, th, tconf, tcls, obj_mask, noobj_mask = build_target(
        gtbox, attrs, x.shape[2])

    obj_mask_expand = np.tile(
        np.expand_dims(obj_mask, 4), (1, 1, 1, 1, int(attrs['class_num'])))
    loss_x = mse(pred_x * obj_mask, tx * obj_mask, obj_mask.sum())
    loss_y = mse(pred_y * obj_mask, ty * obj_mask, obj_mask.sum())
    loss_w = mse(pred_w * obj_mask, tw * obj_mask, obj_mask.sum())
    loss_h = mse(pred_h * obj_mask, th * obj_mask, obj_mask.sum())
    loss_conf_obj = bce(pred_conf * obj_mask, tconf * obj_mask, obj_mask)
    loss_conf_noobj = bce(pred_conf * noobj_mask, tconf * noobj_mask,
                          noobj_mask)
    loss_class = bce(pred_cls * obj_mask_expand, tcls * obj_mask_expand,
                     obj_mask_expand)
    # print "loss_x: ", loss_x
    # print "loss_y: ", loss_y
    # print "loss_w: ", loss_w
    # print "loss_h: ", loss_h
    # print "loss_conf_obj: ", loss_conf_obj
    # print "loss_conf_noobj: ", loss_conf_noobj
    # print "loss_class: ", loss_class

    return loss_x + loss_y + loss_w + loss_h + loss_conf_obj + loss_conf_noobj + loss_class


class TestYolov3LossOp(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'yolov3_loss'
        x = np.random.random(size=self.x_shape).astype('float32')
        gtbox = np.random.random(size=self.gtbox_shape).astype('float32')
        gtbox[:, :, 0] = np.random.randint(0, self.class_num,
                                           self.gtbox_shape[:2])

        self.attrs = {
            "img_height": self.img_height,
            "anchors": self.anchors,
            "class_num": self.class_num,
            "ignore_thresh": self.ignore_thresh,
        }

        self.inputs = {'X': x, 'GTBox': gtbox}
        self.outputs = {'Loss': np.array([YoloV3Loss(x, gtbox, self.attrs)])}
        print self.outputs

    def test_check_output(self):
        self.check_output(atol=1e-3)

    # def test_check_grad_normal(self):
    #     self.check_grad(['X', 'Grid'], 'Output', max_relative_error=0.61)

    def initTestCase(self):
        self.img_height = 608
        self.anchors = [10, 13, 16, 30, 33, 23]
        self.class_num = 10
        self.ignore_thresh = 0.5
        self.x_shape = (5, len(self.anchors) / 2 * (5 + self.class_num), 7, 7)
        self.gtbox_shape = (5, 10, 5)


if __name__ == "__main__":
    unittest.main()
