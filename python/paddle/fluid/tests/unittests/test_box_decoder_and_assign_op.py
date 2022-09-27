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
import sys
import math
from op_test import OpTest


def box_decoder_and_assign(deltas, weights, boxes, box_score, box_clip):
    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] * wx
    dy = deltas[:, 1::4] * wy
    dw = deltas[:, 2::4] * ww
    dh = deltas[:, 3::4] * wh
    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, box_clip)
    dh = np.minimum(dh, box_clip)
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    output_assign_box = []
    for ino in range(len(pred_boxes)):
        rank = np.argsort(-box_score[ino])
        maxidx = rank[0]
        if maxidx == 0:
            maxidx = rank[1]
        beg_pos = maxidx * 4
        end_pos = maxidx * 4 + 4
        output_assign_box.append(pred_boxes[ino, beg_pos:end_pos])
    output_assign_box = np.array(output_assign_box)

    return pred_boxes, output_assign_box


class TestBoxDecoderAndAssignOpWithLoD(OpTest):

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_decoder_and_assign"
        lod = [[4, 8, 8]]
        num_classes = 10
        prior_box = np.random.random((20, 4)).astype('float32')
        prior_box_var = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        target_box = np.random.random((20, 4 * num_classes)).astype('float32')
        box_score = np.random.random((20, num_classes)).astype('float32')
        box_clip = 4.135
        output_box, output_assign_box = box_decoder_and_assign(
            target_box, prior_box_var, prior_box, box_score, box_clip)

        self.inputs = {
            'PriorBox': (prior_box, lod),
            'PriorBoxVar': prior_box_var,
            'TargetBox': (target_box, lod),
            'BoxScore': (box_score, lod),
        }
        self.attrs = {'box_clip': box_clip}
        self.outputs = {
            'DecodeBox': output_box,
            'OutputAssignBox': output_assign_box
        }


if __name__ == '__main__':
    unittest.main()
