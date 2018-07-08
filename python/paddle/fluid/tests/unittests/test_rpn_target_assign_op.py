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
import paddle.fluid.core as core
from op_test import OpTest


def rpn_target_assign(iou, rpn_batch_size_per_im, rpn_positive_overlap,
                      rpn_negative_overlap, fg_fraction):
    iou = np.transpose(iou)
    anchor_to_gt_max = iou.max(axis=1)
    gt_to_anchor_argmax = iou.argmax(axis=0)
    gt_to_anchor_max = iou[gt_to_anchor_argmax, np.arange(iou.shape[1])]
    anchors_with_max_overlap = np.where(iou == gt_to_anchor_max)[0]

    tgt_lbl = np.ones((iou.shape[0], ), dtype=np.int32) * -1
    tgt_lbl[anchors_with_max_overlap] = 1
    tgt_lbl[anchor_to_gt_max >= rpn_positive_overlap] = 1

    num_fg = int(fg_fraction * rpn_batch_size_per_im)
    fg_inds = np.where(tgt_lbl == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        tgt_lbl[disable_inds] = -1
    fg_inds = np.where(tgt_lbl == 1)[0]

    num_bg = rpn_batch_size_per_im - np.sum(tgt_lbl == 1)
    bg_inds = np.where(anchor_to_gt_max < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
        tgt_lbl[enable_inds] = 0
    bg_inds = np.where(tgt_lbl == 0)[0]

    loc_index = fg_inds
    score_index = np.hstack((fg_inds, bg_inds))
    tgt_lbl = np.expand_dims(tgt_lbl, axis=1)
    return loc_index, score_index, tgt_lbl


class TestRpnTargetAssignOp(OpTest):
    def setUp(self):
        iou = np.random.random((10, 8)).astype("float32")
        self.op_type = "rpn_target_assign"
        self.inputs = {'DistMat': iou}
        self.attrs = {
            'rpn_batch_size_per_im': 256,
            'rpn_positive_overlap': 0.95,
            'rpn_negative_overlap': 0.3,
            'fg_fraction': 0.25,
            'fix_seed': True
        }
        loc_index, score_index, tgt_lbl = rpn_target_assign(iou, 256, 0.95, 0.3,
                                                            0.25)
        self.outputs = {
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': tgt_lbl,
        }

    def test_check_output(self):
        self.check_output()


class TestRpnTargetAssignOp2(OpTest):
    def setUp(self):
        iou = np.random.random((10, 20)).astype("float32")
        self.op_type = "rpn_target_assign"
        self.inputs = {'DistMat': iou}
        self.attrs = {
            'rpn_batch_size_per_im': 128,
            'rpn_positive_overlap': 0.5,
            'rpn_negative_overlap': 0.5,
            'fg_fraction': 0.5,
            'fix_seed': True
        }
        loc_index, score_index, tgt_lbl = rpn_target_assign(iou, 128, 0.5, 0.5,
                                                            0.5)
        self.outputs = {
            'LocationIndex': loc_index,
            'ScoreIndex': score_index,
            'TargetLabel': tgt_lbl,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
