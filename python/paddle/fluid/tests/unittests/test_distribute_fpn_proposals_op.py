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

from __future__ import print_function

import unittest
import numpy as np
import math
import sys
from op_test import OpTest


class TestDistributeFPNProposalsOp(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.rois_fpn, self.rois_idx_restore = self.calc_rois_distribute()
        self.inputs = {'FpnRois': (self.rois[:, 1:5], self.rois_lod)}
        self.attrs = {
            'max_level': self.roi_max_level,
            'min_level': self.roi_min_level,
            'refer_scale': self.canonical_scale,
            'refer_level': self.canonical_level
        }
        output = [('out%d' % i, self.rois_fpn[i])
                  for i in range(len(self.rois_fpn))]
        self.outputs = {
            'MultiFpnRois': output,
            'RestoreIndex': self.rois_idx_restore
        }

    def init_test_case(self):
        self.roi_max_level = 5
        self.roi_min_level = 2
        self.canonical_scale = 224
        self.canonical_level = 4
        self.images_shape = [512, 512]

    def boxes_area(self, boxes):
        w = (boxes[:, 2] - boxes[:, 0] + 1)
        h = (boxes[:, 3] - boxes[:, 1] + 1)
        areas = w * h
        assert np.all(areas >= 0), 'Negative areas founds'
        return areas

    def map_rois_to_fpn_levels(self, rois, lvl_min, lvl_max):
        s = np.sqrt(self.boxes_area(rois))
        s0 = self.canonical_scale
        lvl0 = self.canonical_level
        target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))
        target_lvls = np.clip(target_lvls, lvl_min, lvl_max)
        return target_lvls

    def get_sub_lod(self, sub_lvl):
        sub_lod = []
        max_batch_id = sub_lvl[-1]
        for i in range(max_batch_id.astype(np.int32) + 1):
            sub_lod.append(np.where(sub_lvl == i)[0].size)
        return sub_lod

    def add_multilevel_roi(self, rois, target_lvls, lvl_min, lvl_max):
        rois_idx_order = np.empty((0, ))
        rois_fpn = []
        for lvl in range(lvl_min, lvl_max + 1):
            idx_lvl = np.where(target_lvls == lvl)[0]
            if len(idx_lvl) == 0:
                rois_fpn.append((np.empty(shape=(0, 4)), [[0, 0]]))
                continue
            sub_lod = self.get_sub_lod(rois[idx_lvl, 0])
            rois_fpn.append((rois[idx_lvl, 1:], [sub_lod]))
            rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_idx_restore = np.argsort(rois_idx_order).astype(
            np.int32, copy=False)
        return rois_fpn, rois_idx_restore

    def calc_rois_distribute(self):
        lvl_min = self.roi_min_level
        lvl_max = self.roi_max_level
        target_lvls = self.map_rois_to_fpn_levels(self.rois[:, 1:5], lvl_min,
                                                  lvl_max)
        rois_fpn, rois_idx_restore = self.add_multilevel_roi(
            self.rois, target_lvls, lvl_min, lvl_max)
        return rois_fpn, rois_idx_restore

    def make_rois(self):
        self.rois_lod = [[100, 200]]
        rois = []
        lod = self.rois_lod[0]
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
        self.rois = np.array(rois).astype("float32")

    def setUp(self):
        self.op_type = "distribute_fpn_proposals"
        self.set_data()

    def test_check_output(self):
        self.check_output()
