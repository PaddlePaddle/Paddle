#    Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class TestCollectFPNProposalstOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.scores_input = [
            (f'y{i}', (self.scores[i].reshape(-1, 1), self.rois_lod[i]))
            for i in range(self.num_level)
        ]
        self.rois, self.lod = self.calc_rois_collect()
        inputs_x = [
            (f'x{i}', (self.roi_inputs[i][:, 1:], self.rois_lod[i]))
            for i in range(self.num_level)
        ]
        self.inputs = {
            'MultiLevelRois': inputs_x,
            "MultiLevelScores": self.scores_input,
            'MultiLevelRoIsNum': [],
        }
        self.attrs = {
            'post_nms_topN': self.post_nms_top_n,
        }
        self.outputs = {
            'FpnRois': (self.rois, [self.lod]),
            'RoisNum': np.array(self.lod).astype('int32'),
        }

    def init_test_case(self):
        self.post_nms_top_n = 20
        self.images_shape = [100, 100]

    def resort_roi_by_batch_id(self, rois):
        batch_id_list = rois[:, 0]
        batch_size = int(batch_id_list.max())
        sorted_rois = []
        new_lod = []
        for batch_id in range(batch_size + 1):
            sub_ind = np.where(batch_id_list == batch_id)[0]
            sub_rois = rois[sub_ind, 1:]
            sorted_rois.append(sub_rois)
            new_lod.append(len(sub_rois))
        new_rois = np.concatenate(sorted_rois)
        return new_rois, new_lod

    def calc_rois_collect(self):
        roi_inputs = np.concatenate(self.roi_inputs)
        scores = np.concatenate(self.scores)
        inds = np.argsort(-scores)[: self.post_nms_top_n]
        rois = roi_inputs[inds, :]
        new_rois, new_lod = self.resort_roi_by_batch_id(rois)
        return new_rois, new_lod

    def make_rois(self):
        self.num_level = 4
        self.roi_inputs = []
        self.scores = []
        self.rois_lod = [[[20, 10]], [[30, 20]], [[20, 30]], [[10, 10]]]
        for lvl in range(self.num_level):
            rois = []
            scores_pb = []
            lod = self.rois_lod[lvl][0]
            bno = 0
            for roi_num in lod:
                for i in range(roi_num):
                    xywh = np.random.rand(4)
                    xy1 = xywh[0:2] * 20
                    wh = xywh[2:4] * (self.images_shape - xy1)
                    xy2 = xy1 + wh
                    roi = [bno, xy1[0], xy1[1], xy2[0], xy2[1]]
                    rois.append(roi)
                bno += 1
                scores_pb.extend(list(np.random.uniform(0.0, 1.0, roi_num)))
            rois = np.array(rois).astype("float32")
            self.roi_inputs.append(rois)
            scores_pb = np.array(scores_pb).astype("float32")
            self.scores.append(scores_pb)

    def setUp(self):
        self.op_type = "collect_fpn_proposals"
        self.set_data()

    def test_check_output(self):
        self.check_output(check_dygraph=False)


class TestCollectFPNProposalstOpWithRoisNum(TestCollectFPNProposalstOp):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.scores_input = [
            (f'y{i}', (self.scores[i].reshape(-1, 1), self.rois_lod[i]))
            for i in range(self.num_level)
        ]
        self.rois, self.lod = self.calc_rois_collect()
        inputs_x = [
            (f'x{i}', (self.roi_inputs[i][:, 1:], self.rois_lod[i]))
            for i in range(self.num_level)
        ]
        rois_num_per_level = [
            (f'rois{i}', np.array(self.rois_lod[i][0]).astype('int32'))
            for i in range(self.num_level)
        ]

        self.inputs = {
            'MultiLevelRois': inputs_x,
            "MultiLevelScores": self.scores_input,
            'MultiLevelRoIsNum': rois_num_per_level,
        }
        self.attrs = {
            'post_nms_topN': self.post_nms_top_n,
        }
        self.outputs = {
            'FpnRois': (self.rois, [self.lod]),
            'RoisNum': np.array(self.lod).astype('int32'),
        }


if __name__ == '__main__':
    unittest.main()
