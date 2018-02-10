#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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
import random
from op_test import OpTest


def gen_match_and_neg_indices(num_prior, gt_lod, neg_lod):
    if len(gt_lod) != len(neg_lod):
        raise AssertionError("The input arguments are illegal.")

    batch_size = len(gt_lod) - 1

    match_indices = -1 * np.ones((batch_size, num_prior)).astype('int32')
    neg_indices = np.zeros((neg_lod[-1], 1)).astype('int32')

    for n in range(batch_size):
        gt_num = gt_lod[n + 1] - gt_lod[n]
        ids = random.sample([i for i in range(num_prior)], gt_num)
        match_indices[n, ids] = [i for i in range(gt_num)]

        ret_ids = set([i for i in range(num_prior)]) - set(ids)
        s = neg_lod[n]
        e = neg_lod[n + 1]
        l = e - s
        neg_ids = random.sample(ret_ids, l)
        neg_indices[s:e, :] = np.array(neg_ids).astype('int32').reshape(l, 1)

    return match_indices, neg_indices


def target_assign(encoded_box, gt_label, match_indices, neg_indices, gt_lod,
                  neg_lod, background_label):
    batch_size, num_prior = match_indices.shape

    # init target bbox
    trg_box = np.zeros((batch_size, num_prior, 4)).astype('float32')
    # init weight for target bbox
    trg_box_wt = np.zeros((batch_size, num_prior, 1)).astype('float32')
    # init target label
    trg_label = np.ones((batch_size, num_prior, 1)).astype('int32')
    trg_label = trg_label * background_label
    # init weight for target label
    trg_label_wt = np.zeros((batch_size, num_prior, 1)).astype('float32')

    for i in range(batch_size):
        cur_indices = match_indices[i]
        col_ids = np.where(cur_indices > -1)
        col_val = cur_indices[col_ids]

        gt_start = gt_lod[i]
        # target bbox
        for v, c in zip(col_val + gt_start, col_ids[0].tolist()):
            trg_box[i][c][:] = encoded_box[v][c][:]

        # weight for target bbox
        trg_box_wt[i][col_ids] = 1.0

        trg_label[i][col_ids] = gt_label[col_val + gt_start]

        trg_label_wt[i][col_ids] = 1.0
        # set target label weight to 1.0 for the negative samples
        neg_ids = neg_indices[neg_lod[i]:neg_lod[i + 1]]
        trg_label_wt[i][neg_ids] = 1.0

    return trg_box, trg_box_wt, trg_label, trg_label_wt


class TestTargetAssginOp(OpTest):
    def setUp(self):
        self.op_type = "target_assign"

        num_prior = 120
        num_class = 21
        gt_lod = [0, 5, 11, 23]
        neg_lod = [0, 4, 7, 13]
        batch_size = len(gt_lod) - 1
        num_gt = gt_lod[-1]
        background_label = 0

        encoded_box = np.random.random((num_gt, num_prior, 4)).astype('float32')
        gt_label = np.random.randint(
            num_class, size=(num_gt, 1)).astype('int32')
        match_indices, neg_indices = gen_match_and_neg_indices(num_prior,
                                                               gt_lod, neg_lod)
        trg_box, trg_box_wt, trg_label, trg_label_wt = target_assign(
            encoded_box, gt_label, match_indices, neg_indices, gt_lod, neg_lod,
            background_label)

        self.inputs = {
            'EncodedGTBBox': (encoded_box, [gt_lod]),
            'GTScoreLabel': (gt_label, [gt_lod]),
            'MatchIndices': (match_indices),
            'NegIndices': (neg_indices, [neg_lod]),
        }
        self.attrs = {'background_label': background_label}
        self.outputs = {
            'PredBBoxLabel': (trg_box),
            'PredBBoxWeight': (trg_box_wt),
            'PredScoreLabel': (trg_label),
            'PredScoreWeight': (trg_label_wt),
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
