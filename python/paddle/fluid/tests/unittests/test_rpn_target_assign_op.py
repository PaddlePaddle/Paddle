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
import paddle.fluid.core as core
from op_test import OpTest
from test_anchor_generator_op import anchor_generator_in_python
from test_generate_proposal_labels import _generate_groundtruth
from test_generate_proposal_labels import _bbox_overlaps, _box_to_delta


def rpn_target_assign(gt_anchor_iou, rpn_batch_size_per_im,
                      rpn_positive_overlap, rpn_negative_overlap, fg_fraction):
    iou = np.transpose(gt_anchor_iou)
    anchor_to_gt_max = iou.max(axis=1)
    anchor_to_gt_argmax = iou.argmax(axis=1)

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
    tgt_lbl[bg_inds] = 0
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
        tgt_lbl[enable_inds] = 0
    bg_inds = np.where(tgt_lbl == 0)[0]
    tgt_lbl[bg_inds] = 0

    loc_index = fg_inds
    score_index = np.hstack((fg_inds, bg_inds))
    tgt_lbl = np.expand_dims(tgt_lbl, axis=1)

    gt_inds = anchor_to_gt_argmax[fg_inds]

    return loc_index, score_index, tgt_lbl, gt_inds


def get_anchor(n, c, h, w):
    input_feat = np.random.random((n, c, h, w)).astype('float32')
    anchors, _ = anchor_generator_in_python(
        input_feat=input_feat,
        anchor_sizes=[32., 64.],
        aspect_ratios=[0.5, 1.0],
        variances=[1.0, 1.0, 1.0, 1.0],
        stride=[16.0, 16.0],
        offset=0.5)
    return anchors


def rpn_blob(anchor, gt_boxes, iou, lod, rpn_batch_size_per_im,
             rpn_positive_overlap, rpn_negative_overlap, fg_fraction):

    loc_indexes = []
    score_indexes = []
    tmp_tgt_labels = []
    tgt_bboxes = []
    anchor_num = anchor.shape[0]

    batch_size = len(lod) - 1
    for i in range(batch_size):
        b, e = lod[i], lod[i + 1]
        iou_slice = iou[b:e, :]
        bboxes_slice = gt_boxes[b:e, :]

        loc_idx, score_idx, tgt_lbl, gt_inds = rpn_target_assign(
            iou_slice, rpn_batch_size_per_im, rpn_positive_overlap,
            rpn_negative_overlap, fg_fraction)

        fg_bboxes = bboxes_slice[gt_inds]
        fg_anchors = anchor[loc_idx]
        box_deltas = _box_to_delta(fg_anchors, fg_bboxes, [1., 1., 1., 1.])

        if i == 0:
            loc_indexes = loc_idx
            score_indexes = score_idx
            tmp_tgt_labels = tgt_lbl
            tgt_bboxes = box_deltas
        else:
            loc_indexes = np.concatenate(
                [loc_indexes, loc_idx + i * anchor_num])
            score_indexes = np.concatenate(
                [score_indexes, score_idx + i * anchor_num])
            tmp_tgt_labels = np.concatenate([tmp_tgt_labels, tgt_lbl])
            tgt_bboxes = np.vstack([tgt_bboxes, box_deltas])

    tgt_labels = tmp_tgt_labels[score_indexes]
    return loc_indexes, score_indexes, tgt_bboxes, tgt_labels


class TestRpnTargetAssignOp(OpTest):
    def setUp(self):
        n, c, h, w = 2, 4, 14, 14
        anchor = get_anchor(n, c, h, w)
        gt_num = 10
        anchor = anchor.reshape(-1, 4)
        anchor_num = anchor.shape[0]

        im_shapes = [[64, 64], [64, 64]]
        gt_box, lod = _generate_groundtruth(im_shapes, 3, 4)
        bbox = np.vstack([v['boxes'] for v in gt_box])

        iou = _bbox_overlaps(bbox, anchor)

        anchor = anchor.astype('float32')
        bbox = bbox.astype('float32')
        iou = iou.astype('float32')

        loc_index, score_index, tgt_bbox, tgt_lbl = rpn_blob(
            anchor, bbox, iou, [0, 4, 8], 25600, 0.95, 0.03, 0.25)

        self.op_type = "rpn_target_assign"
        self.inputs = {
            'Anchor': anchor,
            'GtBox': (bbox, [[4, 4]]),
            'DistMat': (iou, [[4, 4]]),
        }
        self.attrs = {
            'rpn_batch_size_per_im': 25600,
            'rpn_positive_overlap': 0.95,
            'rpn_negative_overlap': 0.03,
            'fg_fraction': 0.25,
            'fix_seed': True
        }
        self.outputs = {
            'LocationIndex': loc_index.astype('int32'),
            'ScoreIndex': score_index.astype('int32'),
            'TargetBBox': tgt_bbox.astype('float32'),
            'TargetLabel': tgt_lbl.astype('int64'),
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
