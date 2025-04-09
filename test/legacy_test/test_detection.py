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

import contextlib
import unittest

import numpy as np

import paddle
from paddle import base
from paddle.base import core
from paddle.base.framework import Program, program_guard

paddle.enable_static()


@contextlib.contextmanager
def new_program_scope(main=None, startup=None, scope=None):
    prog = main if main else base.Program()
    startup_prog = startup if startup else base.Program()
    scope = scope if scope else base.core.Scope()
    with base.scope_guard(scope):
        with base.program_guard(prog, startup_prog):
            with base.unique_name.guard():
                yield


class LayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = 111

    @classmethod
    def tearDownClass(cls):
        pass

    def _get_place(self, force_to_use_cpu=False):
        # this option for ops that only have cpu kernel
        if force_to_use_cpu:
            return core.CPUPlace()
        else:
            if core.is_compiled_with_cuda():
                return core.CUDAPlace(0)
            return core.CPUPlace()

    @contextlib.contextmanager
    def static_graph(self):
        with new_program_scope():
            paddle.seed(self.seed)
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = base.Executor(self._get_place(force_to_use_cpu))
        exe.run(paddle.static.default_startup_program())
        return exe.run(
            paddle.static.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with base.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            paddle.seed(self.seed)
            yield


class TestGenerateProposals(LayerTest):
    def test_generate_proposals(self):
        scores_np = np.random.rand(2, 3, 4, 4).astype('float32')
        bbox_deltas_np = np.random.rand(2, 12, 4, 4).astype('float32')
        im_info_np = np.array([[8, 8, 0.5], [6, 6, 0.5]]).astype('float32')
        anchors_np = np.reshape(np.arange(4 * 4 * 3 * 4), [4, 4, 3, 4]).astype(
            'float32'
        )
        variances_np = np.ones((4, 4, 3, 4)).astype('float32')

        with self.static_graph():
            scores = paddle.static.data(
                name='scores', shape=[2, 3, 4, 4], dtype='float32'
            )
            bbox_deltas = paddle.static.data(
                name='bbox_deltas', shape=[2, 12, 4, 4], dtype='float32'
            )
            im_info = paddle.static.data(
                name='im_info', shape=[2, 3], dtype='float32'
            )
            anchors = paddle.static.data(
                name='anchors', shape=[4, 4, 3, 4], dtype='float32'
            )
            variances = paddle.static.data(
                name='var', shape=[4, 4, 3, 4], dtype='float32'
            )
            rois, roi_probs, rois_num = paddle.vision.ops.generate_proposals(
                scores,
                bbox_deltas,
                im_info[:2],
                anchors,
                variances,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True,
            )
            (
                rois_stat,
                roi_probs_stat,
                rois_num_stat,
            ) = self.get_static_graph_result(
                feed={
                    'scores': scores_np,
                    'bbox_deltas': bbox_deltas_np,
                    'im_info': im_info_np,
                    'anchors': anchors_np,
                    'var': variances_np,
                },
                fetch_list=[rois, roi_probs, rois_num],
                with_lod=False,
            )

        with self.dynamic_graph():
            scores_dy = paddle.to_tensor(scores_np)
            bbox_deltas_dy = paddle.to_tensor(bbox_deltas_np)
            im_info_dy = paddle.to_tensor(im_info_np)
            anchors_dy = paddle.to_tensor(anchors_np)
            variances_dy = paddle.to_tensor(variances_np)
            rois, roi_probs, rois_num = paddle.vision.ops.generate_proposals(
                scores_dy,
                bbox_deltas_dy,
                im_info_dy[:2],
                anchors_dy,
                variances_dy,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True,
            )
            rois_dy = rois.numpy()
            roi_probs_dy = roi_probs.numpy()
            rois_num_dy = rois_num.numpy()

        np.testing.assert_array_equal(np.array(rois_stat), rois_dy)
        np.testing.assert_array_equal(np.array(roi_probs_stat), roi_probs_dy)
        np.testing.assert_array_equal(np.array(rois_num_stat), rois_num_dy)


class TestDistributeFpnProposals(LayerTest):
    def static_distribute_fpn_proposals(self, rois_np, rois_num_np):
        with self.static_graph():
            rois = paddle.static.data(
                name='rois', shape=[10, 4], dtype='float32'
            )
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32'
            )
            (
                multi_rois,
                restore_ind,
                rois_num_per_level,
            ) = paddle.vision.ops.distribute_fpn_proposals(
                fpn_rois=rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num,
            )
            fetch_list = [*multi_rois, restore_ind, *rois_num_per_level]
            output_stat = self.get_static_graph_result(
                feed={'rois': rois_np, 'rois_num': rois_num_np},
                fetch_list=fetch_list,
                with_lod=True,
            )
            output_stat_np = []
            for output in output_stat:
                output_np = np.array(output)
                if len(output_np) > 0:
                    output_stat_np.append(output_np)
        return output_stat_np

    def dynamic_distribute_fpn_proposals(self, rois_np, rois_num_np):
        with self.dynamic_graph():
            rois_dy = paddle.to_tensor(rois_np)
            rois_num_dy = paddle.to_tensor(rois_num_np)
            (
                multi_rois_dy,
                restore_ind_dy,
                rois_num_per_level_dy,
            ) = paddle.vision.ops.distribute_fpn_proposals(
                fpn_rois=rois_dy,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num_dy,
            )
            print(type(multi_rois_dy))
            output_dy = [*multi_rois_dy, restore_ind_dy, *rois_num_per_level_dy]
            output_dy_np = []
            for output in output_dy:
                output_np = output.numpy()
                if len(output_np) > 0:
                    output_dy_np.append(output_np)
        return output_dy_np

    def test_distribute_fpn_proposals(self):
        rois_np = np.random.rand(10, 4).astype('float32')
        rois_num_np = np.array([4, 6]).astype('int32')

        output_stat_np = self.static_distribute_fpn_proposals(
            rois_np, rois_num_np
        )
        output_dy_np = self.dynamic_distribute_fpn_proposals(
            rois_np, rois_num_np
        )

        for res_stat, res_dy in zip(output_stat_np, output_dy_np):
            np.testing.assert_array_equal(res_stat, res_dy)

    def test_distribute_fpn_proposals_error(self):
        program = Program()
        with program_guard(program):
            fpn_rois = paddle.static.data(
                name='data_error', shape=[10, 4], dtype='int32'
            )
            rois_num = paddle.static.data(
                name='rois_num', shape=[None], dtype='int32'
            )
            self.assertRaises(
                TypeError,
                paddle.vision.ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num,
            )

    def test_distribute_fpn_proposals_error2(self):
        program = Program()
        with program_guard(program):
            fpn_rois = paddle.static.data(
                name='min_max_level_error1',
                shape=[10, 4],
                dtype='float32',
            )
            self.assertRaises(
                AssertionError,
                paddle.vision.ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=0,
                max_level=-1,
                refer_level=4,
                refer_scale=224,
            )

    def test_distribute_fpn_proposals_error3(self):
        program = Program()
        with program_guard(program):
            fpn_rois = paddle.static.data(
                name='min_max_level_error2',
                shape=[10, 4],
                dtype='float32',
            )
            self.assertRaises(
                AssertionError,
                paddle.vision.ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=2,
                refer_level=4,
                refer_scale=224,
            )

    def test_distribute_fpn_proposals_error4(self):
        program = Program()
        with program_guard(program):
            fpn_rois = paddle.static.data(
                name='min_max_level_error3',
                shape=[10, 4],
                dtype='float32',
            )
            self.assertRaises(
                AssertionError,
                paddle.vision.ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=500,
                refer_level=4,
                refer_scale=224,
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
