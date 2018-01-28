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


class TestTargetLocationAssginOp(OpTest):
    def set_data(self):
        self.init_test_case()

        self.inputs = {
            'Loc': self.loc_predic,
            'GTBoxes': (self.encode_gt_boxes, self.encode_gt_boxes_lod),
            'MatchIndices': self.match_indices,
            'PriorBoxes': self.prior_boxes,
            'PriorVariances': self.prior_variance
        }

        self.attrs = {
            'encode_variance_in_target': self.encode_variance_in_target
        }

        self.outputs = {
            'LocGT': self.loc_gt_data,
            'LocPred': self.loc_pred_data
        }

    def init_test_case(self):
        self.batch_size = 10
        self.prior_num = 10
        self.encode_gt_boxes_num = 100
        self.encode_gt_boxes_lod = [[0, 5, 15, 30, 40, 50, 60, 70, 80, 90, 100]]
        self.encode_variance_in_target = False

        self.init_input_data()
        self.loc_gt_data, self.loc_pred_data = self.calc_location_assign()

    def init_input_data(self):
        def make_box(init=0.0):
            xmin = random.uniform(init, 1.0)
            ymin = random.uniform(init, 1.0)
            xmax = random.uniform(xmin, 1.0)
            ymax = random.uniform(ymin, 1.0)
            return [xmin, ymin, xmax, ymax]

        # [batch_size, prior_num, 4]
        self.loc_predic = np.zeros(
            (self.batch_size, self.prior_num, 4)).astype('float32')

        # [encode_gt_boxes_num, prior_num, 4]
        self.encode_gt_boxes = np.zeros(
            (self.encode_gt_boxes_num, self.prior_num, 4)).astype('float32')

        # match_indices[n, p] = gt_box_index
        self.match_indices = np.zeros(
            (self.batch_size, self.prior_num)).astype('int32')

        for n in range(self.batch_size):
            gt_start = self.encode_gt_boxes_lod[0][n]
            gt_end = self.encode_gt_boxes_lod[0][n + 1]
            gt_num = gt_end - gt_start
            for p in range(self.prior_num):
                self.loc_predic[n, p, :] = make_box()
                self.match_indices[n, p] = random.randint(-1, gt_num - 1)

        for g in range(self.encode_gt_boxes_num):
            for p in range(self.prior_num):
                self.encode_gt_boxes[g, p, :] = make_box()

        # shape = [prior_num, 4]
        self.prior_boxes = np.zeros((self.prior_num, 4)).astype('float32')
        for p in range(self.prior_num):
            self.prior_boxes[p, :] = make_box()

        self.prior_variance = np.tile([0.1, 0.1, 0.2, 0.2],
                                      (self.prior_num, 1)).astype('float32')

    def encode_box(self,
                   prior_box,
                   prior_variance,
                   gt_box,
                   encode_variance_in_target=False):
        prior_xmin, prior_ymin, prior_xmax, prior_ymax = prior_box[:]
        prior_width = prior_xmax - prior_xmin
        prior_height = prior_ymax - prior_ymin
        prior_center_x = (prior_xmin + prior_xmax) / 2.
        prior_center_y = (prior_ymin + prior_ymax) / 2.

        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_box[:]
        gt_width = gt_xmax - gt_xmin
        gt_height = gt_ymax - gt_ymin
        gt_center_x = (gt_xmin + gt_xmax) / 2.
        gt_center_y = (gt_ymin + gt_ymax) / 2.

        if encode_variance_in_target:
            encode_xmin = (gt_center_x - prior_center_x) / prior_width
            encode_ymin = (gt_center_y - prior_center_y) / prior_height
            encode_xmax = math.log(gt_width / prior_width)
            encode_ymax = math.log(gt_height / prior_height)
        else:
            encode_xmin = \
                (gt_center_x - prior_center_x) / prior_width / prior_variance[0]
            encode_ymin = \
                (gt_center_y - prior_center_y) / prior_height / prior_variance[1]
            encode_xmax = \
                math.log(gt_width / prior_width) / prior_variance[2]
            encode_ymax = \
                math.log(gt_height / prior_height) / prior_variance[3]

        return [encode_xmin, encode_ymin, encode_xmax, encode_ymax]

    def calc_location_assign(self, encode_variance_in_target=True):
        loc_gt_data = []
        loc_pred_data = []
        target_lod = [0]
        count = 0
        for i in range(self.batch_size):
            for j in range(self.prior_num):
                idx = self.match_indices[i, j]
                if idx == -1: continue
                gt_start = self.encode_gt_boxes_lod[0][i]
                idx = idx + gt_start
                gt_box = self.encode_gt_boxes[idx, j, :]

                gt_encode = self.encode_box(self.prior_boxes[j],
                                            self.prior_variance[j], gt_box,
                                            encode_variance_in_target)

                loc_pred = self.loc_predic[i, j, :]
                if (encode_variance_in_target):
                    for k in range(4):
                        loc_pred[k] /= self.prior_variance[j, k]
                        gt_encode[k] /= self.prior_variance[j, k]

                loc_gt_data.append(gt_encode)
                loc_pred_data.append(loc_pred)
                count = count + 1
            target_lod.append(count)
        return (np.array(loc_gt_data).astype('float32'), [target_lod]), \
                (np.array(loc_pred_data).astype('float32'), [target_lod])

    def setUp(self):
        self.op_type = "target_location_assign"
        self.set_data()

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
