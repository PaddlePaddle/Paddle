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

from __future__ import print_function

import unittest
import numpy as np
import sys
import math
from op_test import OpTest

def box_decoder_and_assign(prior_box, prior_box_var, target_box, box_score, box_clip):
    proposals = np.zeros_like(target_box, dtype=np.float32)
    prior_box_loc = np.zeros_like(prior_box, dtype=np.float32)
    prior_box_loc[:, 0] = prior_box[:, 2] - prior_box[:, 0] + 1.
    prior_box_loc[:, 1] = prior_box[:, 3] - prior_box[:, 1] + 1.
    prior_box_loc[:, 2] = (prior_box[:, 2] + prior_box[:, 0]) / 2
    prior_box_loc[:, 3] = (prior_box[:, 3] + prior_box[:, 1]) / 2
    pred_bbox = np.zeros_like(target_box, dtype=np.float32)
    for i in range(prior_box.shape[0]):
        dw = np.minimum(prior_box_var[2] * target_box[i, 2::4], box_clip)
        dh = np.minimum(prior_box_var[3] * target_box[i, 3::4], box_clip)
        pred_bbox[i, 0::4] = prior_box_var[0] * target_box[
            i, 0::4] * prior_box_loc[i, 0] + prior_box_loc[i, 2]
        pred_bbox[i, 1::4] = prior_box_var[1] * target_box[
            i, 1::4] * prior_box_loc[i, 1] + prior_box_loc[i, 3]
        pred_bbox[i, 2::4] = np.exp(dw) * prior_box_loc[i, 0]
        pred_bbox[i, 3::4] = np.exp(dh) * prior_box_loc[i, 1]
    proposals[:, 0::4] = pred_bbox[:, 0::4] - pred_bbox[:, 2::4] / 2
    proposals[:, 1::4] = pred_bbox[:, 1::4] - pred_bbox[:, 3::4] / 2
    proposals[:, 2::4] = pred_bbox[:, 0::4] + pred_bbox[:, 2::4] / 2 - 1
    proposals[:, 3::4] = pred_bbox[:, 1::4] + pred_bbox[:, 3::4] / 2 - 1
    output_assign_box = []
    for ino in range(len(proposals)):
        rank = np.argsort(-box_score[ino])
        maxidx = rank[0]
        if maxidx == 0:
            maxidx = rank[1]
        beg_pos = maxidx * 4
        end_pos = maxidx * 4 + 4
        output_assign_box.append(proposals[ino, beg_pos:end_pos])
    output_assign_box = np.array(output_assign_box)
    return proposals, output_assign_box

class TestBoxDecoderAndAssignOpWithLoD(OpTest):
    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = "box_decoder_and_assign"
        lod = [[4, 8, 8]]
        num_classes = 10
        prior_box = np.random.random((20, 4)).astype('float32')
        prior_box_var = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
        target_box = np.random.random((20, 4*num_classes)).astype('float32')
        box_score = np.random.random((20, num_classes)).astype('float32')
        box_clip = 4.135
        output_box, output_assign_box = box_decoder_and_assign(prior_box, prior_box_var, target_box, box_score, box_clip)        

        self.inputs = {
            'PriorBox': (prior_box, lod),
            'PriorBoxVar': prior_box_var,
            'TargetBox': (target_box, lod),
            'BoxScore': (box_score, lod),
        }
        self.attrs = {'box_clip': box_clip}
        self.outputs = {'OutputBox': output_box, 'OutputAssignBox':output_assign_box}

if __name__ == '__main__':
    unittest.main()
