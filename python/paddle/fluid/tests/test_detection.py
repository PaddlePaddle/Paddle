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
from unittests.test_imperative_base import new_program_scope

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid import core
from paddle.fluid.dygraph import base
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.layers import detection

paddle.enable_static()


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
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield

    def get_static_graph_result(
        self, feed, fetch_list, with_lod=False, force_to_use_cpu=False
    ):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(
            fluid.default_main_program(),
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=(not with_lod),
        )

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
            self._get_place(force_to_use_cpu=force_to_use_cpu)
        ):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield


class TestDetection(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            loc = layers.data(
                name='target_box',
                shape=[2, 10, 4],
                append_batch_size=False,
                dtype='float32',
            )
            scores = layers.data(
                name='scores',
                shape=[2, 10, 20],
                append_batch_size=False,
                dtype='float32',
            )
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv
            )
            out2, index = layers.detection_output(
                scores=scores,
                loc=loc,
                prior_box=pb,
                prior_box_var=pbv,
                return_index=True,
            )
            self.assertIsNotNone(out)
            self.assertIsNotNone(out2)
            self.assertIsNotNone(index)
            self.assertEqual(out.shape[-1], 6)
        print(str(program))

    def test_box_coder_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y,
                code_type='encode_center_size',
            )
            self.assertIsNotNone(bcoder)
        print(str(program))

    def test_box_coder_error(self):
        program = Program()
        with program_guard(program):
            x1 = fluid.data(name='x1', shape=[10, 4], dtype='int32')
            y1 = fluid.data(
                name='y1', shape=[10, 4], dtype='float32', lod_level=1
            )
            x2 = fluid.data(name='x2', shape=[10, 4], dtype='float32')
            y2 = fluid.data(
                name='y2', shape=[10, 4], dtype='int32', lod_level=1
            )

            self.assertRaises(
                TypeError,
                layers.box_coder,
                prior_box=x1,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y1,
                code_type='encode_center_size',
            )
            self.assertRaises(
                TypeError,
                layers.box_coder,
                prior_box=x2,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y2,
                code_type='encode_center_size',
            )


class TestDetectionMAP(unittest.TestCase):
    def test_detection_map(self):
        program = Program()
        with program_guard(program):
            detect_res = layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32',
            )
            label = layers.data(
                name='label',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32',
            )

            map_out = detection.detection_map(detect_res, label, 21)
            self.assertIsNotNone(map_out)
            self.assertEqual(map_out.shape, (1,))
        print(str(program))


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
            scores = fluid.data(
                name='scores', shape=[2, 3, 4, 4], dtype='float32'
            )
            bbox_deltas = fluid.data(
                name='bbox_deltas', shape=[2, 12, 4, 4], dtype='float32'
            )
            im_info = fluid.data(name='im_info', shape=[2, 3], dtype='float32')
            anchors = fluid.data(
                name='anchors', shape=[4, 4, 3, 4], dtype='float32'
            )
            variances = fluid.data(
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
            scores_dy = base.to_variable(scores_np)
            bbox_deltas_dy = base.to_variable(bbox_deltas_np)
            im_info_dy = base.to_variable(im_info_np)
            anchors_dy = base.to_variable(anchors_np)
            variances_dy = base.to_variable(variances_np)
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


class TestMulticlassNMS2(unittest.TestCase):
    def test_multiclass_nms2(self):
        program = Program()
        with program_guard(program):
            bboxes = layers.data(
                name='bboxes', shape=[-1, 10, 4], dtype='float32'
            )
            scores = layers.data(name='scores', shape=[-1, 10], dtype='float32')
            output = fluid.contrib.multiclass_nms2(
                bboxes, scores, 0.3, 400, 200, 0.7
            )
            output2, index = fluid.contrib.multiclass_nms2(
                bboxes, scores, 0.3, 400, 200, 0.7, return_index=True
            )
            self.assertIsNotNone(output)
            self.assertIsNotNone(output2)
            self.assertIsNotNone(index)


class TestDistributeFpnProposals(LayerTest):
    def test_distribute_fpn_proposals(self):
        rois_np = np.random.rand(10, 4).astype('float32')
        rois_num_np = np.array([4, 6]).astype('int32')
        with self.static_graph():
            rois = fluid.data(name='rois', shape=[10, 4], dtype='float32')
            rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')
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
            fetch_list = multi_rois + [restore_ind] + rois_num_per_level
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

        with self.dynamic_graph():
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)
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
            output_dy = multi_rois_dy + [restore_ind_dy] + rois_num_per_level_dy
            output_dy_np = []
            for output in output_dy:
                output_np = output.numpy()
                if len(output_np) > 0:
                    output_dy_np.append(output_np)

        for res_stat, res_dy in zip(output_stat_np, output_dy_np):
            np.testing.assert_array_equal(res_stat, res_dy)

    def test_distribute_fpn_proposals_error(self):
        program = Program()
        with program_guard(program):
            fpn_rois = fluid.data(
                name='data_error', shape=[10, 4], dtype='int32', lod_level=1
            )
            self.assertRaises(
                TypeError,
                paddle.vision.ops.distribute_fpn_proposals,
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
            )


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
