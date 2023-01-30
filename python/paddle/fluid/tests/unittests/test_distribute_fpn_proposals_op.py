# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle


def distribute_fpn_proposals_wrapper(
    fpn_rois,
    rois_num,
    min_level,
    max_level,
    refer_level,
    refer_scale,
    pixel_offset,
):
    return paddle.vision.ops.distribute_fpn_proposals(
        fpn_rois=fpn_rois,
        min_level=min_level,
        max_level=max_level,
        refer_level=refer_level,
        refer_scale=refer_scale,
        rois_num=rois_num,
    )


class TestDistributeFPNProposalsOp(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
import math
import sys
import paddle

from op_test import OpTest


def distribute_fpn_proposals_wrapper(fpn_rois, rois_num, min_level, max_level,
                                     refer_level, refer_scale, pixel_offset):
    return paddle.vision.ops.distribute_fpn_proposals(fpn_rois=fpn_rois,
                                                      min_level=min_level,
                                                      max_level=max_level,
                                                      refer_level=refer_level,
                                                      refer_scale=refer_scale,
                                                      rois_num=rois_num)


class TestDistributeFPNProposalsOp(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.rois_fpn, self.rois_idx_restore = self.calc_rois_distribute()
        self.inputs = {'FpnRois': (self.rois[:, 1:5], self.rois_lod)}
        self.attrs = {
            'max_level': self.roi_max_level,
            'min_level': self.roi_min_level,
            'refer_scale': self.canonical_scale,
            'refer_level': self.canonical_level,
            'pixel_offset': self.pixel_offset,
        }
<<<<<<< HEAD
        output = [
            ('out%d' % i, self.rois_fpn[i]) for i in range(len(self.rois_fpn))
        ]
=======
        output = [('out%d' % i, self.rois_fpn[i])
                  for i in range(len(self.rois_fpn))]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.outputs = {
            'MultiFpnRois': output,
            'RestoreIndex': self.rois_idx_restore.reshape(-1, 1),
        }
        self.python_api = distribute_fpn_proposals_wrapper
        self.python_out_sig = ['MultiFpnRois', 'RestoreIndex']

    def init_test_case(self):
        self.roi_max_level = 5
        self.roi_min_level = 2
        self.canonical_scale = 224
        self.canonical_level = 4
        self.images_shape = [512, 512]
        self.pixel_offset = True

    def boxes_area(self, boxes):
        offset = 1 if self.pixel_offset else 0
<<<<<<< HEAD
        w = boxes[:, 2] - boxes[:, 0] + offset
        h = boxes[:, 3] - boxes[:, 1] + offset
=======
        w = (boxes[:, 2] - boxes[:, 0] + offset)
        h = (boxes[:, 3] - boxes[:, 1] + offset)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        areas = w * h
        assert np.all(areas >= 0), 'Negative areas founds'
        return areas

    def map_rois_to_fpn_levels(self, rois, lvl_min, lvl_max):
        s = np.sqrt(self.boxes_area(rois))
        s0 = self.canonical_scale
        lvl0 = self.canonical_level
        target_lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-8))
        target_lvls = np.clip(target_lvls, lvl_min, lvl_max)
        return target_lvls

    def get_sub_lod(self, sub_lvl):
        sub_lod = [0, 0]
        max_batch_id = sub_lvl[-1]
        for i in range(max_batch_id.astype(np.int32) + 1):
            sub_lod[i] = np.where(sub_lvl == i)[0].size
        return sub_lod

    def add_multilevel_roi(self, rois, target_lvls, lvl_min, lvl_max):
<<<<<<< HEAD
        rois_idx_order = np.empty((0,))
=======
        rois_idx_order = np.empty((0, ))
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        rois_fpn = []
        for lvl in range(lvl_min, lvl_max + 1):
            idx_lvl = np.where(target_lvls == lvl)[0]
            if len(idx_lvl) == 0:
                rois_fpn.append((np.empty(shape=(0, 4)), [[0, 0]]))
                continue
            sub_lod = self.get_sub_lod(rois[idx_lvl, 0])
            rois_fpn.append((rois[idx_lvl, 1:], [sub_lod]))
            rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
<<<<<<< HEAD
        rois_idx_restore = np.argsort(rois_idx_order).astype(
            np.int32, copy=False
        )
=======
        rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32,
                                                             copy=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        return rois_fpn, rois_idx_restore

    def calc_rois_distribute(self):
        lvl_min = self.roi_min_level
        lvl_max = self.roi_max_level
<<<<<<< HEAD
        target_lvls = self.map_rois_to_fpn_levels(
            self.rois[:, 1:5], lvl_min, lvl_max
        )
        rois_fpn, rois_idx_restore = self.add_multilevel_roi(
            self.rois, target_lvls, lvl_min, lvl_max
        )
=======
        target_lvls = self.map_rois_to_fpn_levels(self.rois[:, 1:5], lvl_min,
                                                  lvl_max)
        rois_fpn, rois_idx_restore = self.add_multilevel_roi(
            self.rois, target_lvls, lvl_min, lvl_max)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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


class TestDistributeFPNProposalsOpWithRoisNum(TestDistributeFPNProposalsOp):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def set_data(self):
        self.init_test_case()
        self.make_rois()
        self.rois_fpn, self.rois_idx_restore = self.calc_rois_distribute()
        self.inputs = {
            'FpnRois': (self.rois[:, 1:5], self.rois_lod),
<<<<<<< HEAD
            'RoisNum': np.array(self.rois_lod[0]).astype('int32'),
=======
            'RoisNum': np.array(self.rois_lod[0]).astype('int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.attrs = {
            'max_level': self.roi_max_level,
            'min_level': self.roi_min_level,
            'refer_scale': self.canonical_scale,
            'refer_level': self.canonical_level,
            'pixel_offset': self.pixel_offset,
        }
<<<<<<< HEAD
        output = [
            ('out%d' % i, self.rois_fpn[i]) for i in range(len(self.rois_fpn))
        ]
        rois_num_per_level = [
            ('rois_num%d' % i, np.array(self.rois_fpn[i][1][0]).astype('int32'))
            for i in range(len(self.rois_fpn))
        ]
=======
        output = [('out%d' % i, self.rois_fpn[i])
                  for i in range(len(self.rois_fpn))]
        rois_num_per_level = [('rois_num%d' % i,
                               np.array(self.rois_fpn[i][1][0]).astype('int32'))
                              for i in range(len(self.rois_fpn))]
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        self.outputs = {
            'MultiFpnRois': output,
            'RestoreIndex': self.rois_idx_restore.reshape(-1, 1),
<<<<<<< HEAD
            'MultiLevelRoIsNum': rois_num_per_level,
        }
        self.python_api = distribute_fpn_proposals_wrapper
        self.python_out_sig = [
            'MultiFpnRois',
            'MultiLevelRoIsNum',
            'RestoreIndex',
=======
            'MultiLevelRoIsNum': rois_num_per_level
        }
        self.python_api = distribute_fpn_proposals_wrapper
        self.python_out_sig = [
            'MultiFpnRois', 'MultiLevelRoIsNum', 'RestoreIndex'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        ]


class TestDistributeFPNProposalsOpNoOffset(
<<<<<<< HEAD
    TestDistributeFPNProposalsOpWithRoisNum
):
=======
        TestDistributeFPNProposalsOpWithRoisNum):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_test_case(self):
        self.roi_max_level = 5
        self.roi_min_level = 2
        self.canonical_scale = 224
        self.canonical_level = 4
        self.images_shape = [512, 512]
        self.pixel_offset = False


class TestDistributeFpnProposalsAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        np.random.seed(678)
        self.rois_np = np.random.rand(10, 4).astype('float32')
        self.rois_num_np = np.array([4, 6]).astype('int32')

    def test_dygraph_with_static(self):
        paddle.enable_static()
        rois = paddle.static.data(name='rois', shape=[10, 4], dtype='float32')
<<<<<<< HEAD
        rois_num = paddle.static.data(
            name='rois_num', shape=[None], dtype='int32'
        )
        (
            multi_rois,
            restore_ind,
            rois_num_per_level,
        ) = paddle.vision.ops.distribute_fpn_proposals(
=======
        rois_num = paddle.static.data(name='rois_num',
                                      shape=[None],
                                      dtype='int32')
        multi_rois, restore_ind, rois_num_per_level = paddle.vision.ops.distribute_fpn_proposals(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            fpn_rois=rois,
            min_level=2,
            max_level=5,
            refer_level=4,
            refer_scale=224,
<<<<<<< HEAD
            rois_num=rois_num,
        )
        fetch_list = multi_rois + [restore_ind] + rois_num_per_level

        exe = paddle.static.Executor()
        output_stat = exe.run(
            paddle.static.default_main_program(),
            feed={'rois': self.rois_np, 'rois_num': self.rois_num_np},
            fetch_list=fetch_list,
            return_numpy=False,
        )
=======
            rois_num=rois_num)
        fetch_list = multi_rois + [restore_ind] + rois_num_per_level

        exe = paddle.static.Executor()
        output_stat = exe.run(paddle.static.default_main_program(),
                              feed={
                                  'rois': self.rois_np,
                                  'rois_num': self.rois_num_np
                              },
                              fetch_list=fetch_list,
                              return_numpy=False)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output_stat_np = []
        for output in output_stat:
            output_np = np.array(output)
            if len(output_np) > 0:
                output_stat_np.append(output_np)

        paddle.disable_static()
        rois_dy = paddle.to_tensor(self.rois_np)
        rois_num_dy = paddle.to_tensor(self.rois_num_np)
<<<<<<< HEAD
        (
            multi_rois_dy,
            restore_ind_dy,
            rois_num_per_level_dy,
        ) = paddle.vision.ops.distribute_fpn_proposals(
=======
        multi_rois_dy, restore_ind_dy, rois_num_per_level_dy = paddle.vision.ops.distribute_fpn_proposals(
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            fpn_rois=rois_dy,
            min_level=2,
            max_level=5,
            refer_level=4,
            refer_scale=224,
<<<<<<< HEAD
            rois_num=rois_num_dy,
        )
=======
            rois_num=rois_num_dy)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        output_dy = multi_rois_dy + [restore_ind_dy] + rois_num_per_level_dy
        output_dy_np = []
        for output in output_dy:
            output_np = output.numpy()
            if len(output_np) > 0:
                output_dy_np.append(output_np)

        for res_stat, res_dy in zip(output_stat_np, output_dy_np):
            np.testing.assert_allclose(res_stat, res_dy, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
