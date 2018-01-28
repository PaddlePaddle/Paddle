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
import math
import sys
import random
from op_test import OpTest


class TestTargetConfidenceAssginOp(OpTest):
    def set_data(self):
        self.init_test_case()

        self.inputs = {
            'Conf': self.conf_data,
            'GTLabels': (self.gt_labels, self.gt_labels_lod),
            'MatchIndices': self.match_indices,
            'NegIndices': (self.neg_indices, self.neg_indices_lod)
        }

        self.attrs = {'background_label_id': self.background_label_id}

        self.outputs = {
            'ConfGT': self.conf_gt_data,
            'ConfPred': self.conf_pred_data
        }

    def init_test_case(self):
        self.batch_size = 5
        self.prior_num = 32
        self.cls_num = 1
        self.gt_labels_num = 10
        self.gt_labels_lod = [[0, 2, 5, 7, 9, 10]]
        self.neg_indices_num = 10
        self.neg_indices_lod = [[0, 2, 5, 7, 9, 10]]

        # Only support do_neg_mining=True and loss_type=softmax
        self.do_neg_mining = True
        self.loss_type = 'softmax'  # softmax or logistic
        self.background_label_id = 0

        self.init_input_data()
        self.conf_gt_data, self.conf_pred_data = self.calc_confidence_assign()

    def init_input_data(self):
        # [batch_size, prior_num, cls_num]
        self.conf_data = np.random.random(
            (self.batch_size, self.prior_num, self.cls_num)).astype('float32')

        # [gt_labels_num, 1]
        self.gt_labels = np.random.random_integers(
            0, high=self.cls_num - 1,
            size=(self.gt_labels_num, 1)).astype('int32')

        # match_indices[n, p] = gt_box_index
        self.match_indices = np.zeros(
            (self.batch_size, self.prior_num)).astype('int32')

        self.neg_indices = np.zeros((self.neg_indices_num, 1)).astype('int32')

        for n in range(self.batch_size):
            gt_start = self.gt_labels_lod[0][n]
            gt_end = self.gt_labels_lod[0][n + 1]
            gt_num = gt_end - gt_start
            for p in range(self.prior_num):
                self.match_indices[n, p] = random.randint(-1, gt_num - 1)

            neg_start = self.neg_indices_lod[0][n]
            neg_end = self.neg_indices_lod[0][n + 1]
            for i in range(neg_start, neg_end):
                self.neg_indices[i] = random.randint(0, self.prior_num - 1)

    def calc_confidence_assign(self,
                               do_neg_mining=True,
                               conf_loss_type='softmax'):
        background_label_id = self.background_label_id
        target_lod = [0]
        count = 0

        num_matches = 0
        num_negs = self.neg_indices_num
        for i in range(self.batch_size):
            for j in range(self.prior_num):
                if self.match_indices[i, j] != -1:
                    num_matches += 1
            neg_start = self.neg_indices_lod[0][i]
            neg_end = self.neg_indices_lod[0][i + 1]

        if do_neg_mining:
            num_conf = num_matches + num_negs
        else:
            num_conf = self.batch_size * self.prior_num

        if conf_loss_type == 'softmax':
            conf_gt_data = np.zeros((num_conf, 1)).astype('int32')
            conf_pred_data = np.zeros(
                (num_conf, self.cls_num)).astype('float32')
        elif conf_loss_type == 'logistic':
            conf_gt_data = np.zeros((num_conf, self.cls_num)).astype('int32')
            conf_pred_data = np.zeros(
                (num_conf, self.cls_num)).astype('float32')

        for i in range(self.batch_size):
            for j in range(self.prior_num):
                gt_idx = self.match_indices[i, j]
                if gt_idx == -1: continue
                gt_start = self.gt_labels_lod[0][i]
                gt_idx = gt_idx + gt_start
                gt_label = self.gt_labels[gt_idx]

                if do_neg_mining:
                    idx = count
                else:
                    idx = j

                if conf_loss_type == 'softmax':
                    conf_gt_data[idx] = gt_label
                elif conf_loss_type == 'logistic':
                    conf_gt_data[idx, gt_label] = 1

                if do_neg_mining:
                    conf_pred_data[idx, :] = self.conf_data[i, j, :]
                    count += 1

            # Go to next image.
            if do_neg_mining:
                neg_start = self.neg_indices_lod[0][i]
                neg_end = self.neg_indices_lod[0][i + 1]
                for ne in range(neg_start, neg_end):
                    idx = self.neg_indices[ne]
                    conf_pred_data[count, :] = self.conf_data[i, idx, :]
                    if conf_loss_type == 'softmax':
                        conf_gt_data[count, 0] = background_label_id
                    elif conf_loss_type == 'logistic':
                        conf_gt_data[count, background_label_id] = 1
                    count += 1

            if do_neg_mining:
                target_lod.append(count)

        if do_neg_mining:
            return (conf_gt_data, [target_lod]), (conf_pred_data, [target_lod])
        else:
            return conf_gt_data, conf_pred_data

    def setUp(self):
        self.op_type = "target_confidence_assign"
        self.set_data()

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
