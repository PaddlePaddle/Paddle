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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.layers import detection
from paddle.fluid.framework import Program, program_guard
import unittest
import contextlib
import numpy as np
from unittests.test_imperative_base import new_program_scope
from paddle.fluid.dygraph import base
from paddle.fluid import core
import paddle

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

    def get_static_graph_result(self,
                                feed,
                                fetch_list,
                                with_lod=False,
                                force_to_use_cpu=False):
        exe = fluid.Executor(self._get_place(force_to_use_cpu))
        exe.run(fluid.default_startup_program())
        return exe.run(fluid.default_main_program(),
                       feed=feed,
                       fetch_list=fetch_list,
                       return_numpy=(not with_lod))

    @contextlib.contextmanager
    def dynamic_graph(self, force_to_use_cpu=False):
        with fluid.dygraph.guard(
                self._get_place(force_to_use_cpu=force_to_use_cpu)):
            fluid.default_startup_program().random_seed = self.seed
            fluid.default_main_program().random_seed = self.seed
            yield


class TestDetection(unittest.TestCase):

    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(name='prior_box',
                             shape=[10, 4],
                             append_batch_size=False,
                             dtype='float32')
            pbv = layers.data(name='prior_box_var',
                              shape=[10, 4],
                              append_batch_size=False,
                              dtype='float32')
            loc = layers.data(name='target_box',
                              shape=[2, 10, 4],
                              append_batch_size=False,
                              dtype='float32')
            scores = layers.data(name='scores',
                                 shape=[2, 10, 20],
                                 append_batch_size=False,
                                 dtype='float32')
            out = layers.detection_output(scores=scores,
                                          loc=loc,
                                          prior_box=pb,
                                          prior_box_var=pbv)
            out2, index = layers.detection_output(scores=scores,
                                                  loc=loc,
                                                  prior_box=pb,
                                                  prior_box_var=pbv,
                                                  return_index=True)
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
            bcoder = layers.box_coder(prior_box=x,
                                      prior_box_var=[0.1, 0.2, 0.1, 0.2],
                                      target_box=y,
                                      code_type='encode_center_size')
            self.assertIsNotNone(bcoder)
        print(str(program))

    def test_box_coder_error(self):
        program = Program()
        with program_guard(program):
            x1 = fluid.data(name='x1', shape=[10, 4], dtype='int32')
            y1 = fluid.data(name='y1',
                            shape=[10, 4],
                            dtype='float32',
                            lod_level=1)
            x2 = fluid.data(name='x2', shape=[10, 4], dtype='float32')
            y2 = fluid.data(name='y2',
                            shape=[10, 4],
                            dtype='int32',
                            lod_level=1)

            self.assertRaises(TypeError,
                              layers.box_coder,
                              prior_box=x1,
                              prior_box_var=[0.1, 0.2, 0.1, 0.2],
                              target_box=y1,
                              code_type='encode_center_size')
            self.assertRaises(TypeError,
                              layers.box_coder,
                              prior_box=x2,
                              prior_box_var=[0.1, 0.2, 0.1, 0.2],
                              target_box=y2,
                              code_type='encode_center_size')

    def test_detection_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='y', shape=[4], dtype='float32')
            z = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            iou = layers.iou_similarity(x=x, y=y)
            bcoder = layers.box_coder(prior_box=x,
                                      prior_box_var=y,
                                      target_box=z,
                                      code_type='encode_center_size')
            self.assertIsNotNone(iou)
            self.assertIsNotNone(bcoder)

            matched_indices, matched_dist = layers.bipartite_match(iou)
            self.assertIsNotNone(matched_indices)
            self.assertIsNotNone(matched_dist)

            gt = layers.data(name='gt',
                             shape=[1, 1],
                             dtype='int32',
                             lod_level=1)
            trg, trg_weight = layers.target_assign(gt,
                                                   matched_indices,
                                                   mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

            gt2 = layers.data(name='gt2',
                              shape=[10, 4],
                              dtype='float32',
                              lod_level=1)
            trg, trg_weight = layers.target_assign(gt2,
                                                   matched_indices,
                                                   mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

        print(str(program))

    def test_ssd_loss(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(name='prior_box',
                             shape=[10, 4],
                             append_batch_size=False,
                             dtype='float32')
            pbv = layers.data(name='prior_box_var',
                              shape=[10, 4],
                              append_batch_size=False,
                              dtype='float32')
            loc = layers.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = layers.data(name='gt_box',
                                 shape=[4],
                                 lod_level=1,
                                 dtype='float32')
            gt_label = layers.data(name='gt_label',
                                   shape=[1],
                                   lod_level=1,
                                   dtype='int32')
            loss = layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
            self.assertIsNotNone(loss)
            self.assertEqual(loss.shape[-1], 1)
        print(str(program))


class TestPriorBox(unittest.TestCase):

    def test_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 224, 224]
            images = fluid.layers.data(name='pixel',
                                       shape=data_shape,
                                       dtype='float32')
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.prior_box(input=conv1,
                                        image=images,
                                        min_sizes=[100.0],
                                        aspect_ratios=[1.],
                                        flip=True,
                                        clip=True)
            assert len(box.shape) == 4
            assert box.shape == var.shape
            assert box.shape[3] == 4


class TestPriorBox2(unittest.TestCase):

    def test_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [None, 3, None, None]
            images = fluid.data(name='pixel', shape=data_shape, dtype='float32')
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.prior_box(input=conv1,
                                        image=images,
                                        min_sizes=[100.0],
                                        aspect_ratios=[1.],
                                        flip=True,
                                        clip=True)
            assert len(box.shape) == 4
            assert box.shape == var.shape
            assert box.shape[3] == 4


class TestDensityPriorBox(unittest.TestCase):

    def test_density_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 224, 224]
            images = fluid.layers.data(name='pixel',
                                       shape=data_shape,
                                       dtype='float32')
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.density_prior_box(input=conv1,
                                                image=images,
                                                densities=[3, 4],
                                                fixed_sizes=[50., 60.],
                                                fixed_ratios=[1.0],
                                                clip=True)
            assert len(box.shape) == 4
            assert box.shape == var.shape
            assert box.shape[-1] == 4


class TestAnchorGenerator(unittest.TestCase):

    def test_anchor_generator(self):
        data_shape = [3, 224, 224]
        images = fluid.layers.data(name='pixel',
                                   shape=data_shape,
                                   dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        anchor, var = fluid.layers.anchor_generator(
            input=conv1,
            anchor_sizes=[64, 128, 256, 512],
            aspect_ratios=[0.5, 1.0, 2.0],
            variance=[0.1, 0.1, 0.2, 0.2],
            stride=[16.0, 16.0],
            offset=0.5)
        assert len(anchor.shape) == 4
        assert anchor.shape == var.shape
        assert anchor.shape[3] == 4


class TestGenerateProposalLabels(unittest.TestCase):

    def check_out(self, outs):
        rois = outs[0]
        labels_int32 = outs[1]
        bbox_targets = outs[2]
        bbox_inside_weights = outs[3]
        bbox_outside_weights = outs[4]
        assert rois.shape[1] == 4
        assert rois.shape[0] == labels_int32.shape[0]
        assert rois.shape[0] == bbox_targets.shape[0]
        assert rois.shape[0] == bbox_inside_weights.shape[0]
        assert rois.shape[0] == bbox_outside_weights.shape[0]
        assert bbox_targets.shape[1] == 4 * self.class_nums
        assert bbox_inside_weights.shape[1] == 4 * self.class_nums
        assert bbox_outside_weights.shape[1] == 4 * self.class_nums
        if len(outs) == 6:
            max_overlap_with_gt = outs[5]
            assert max_overlap_with_gt.shape[0] == rois.shape[0]

    def test_generate_proposal_labels(self):
        program = Program()
        with program_guard(program):
            rpn_rois = fluid.data(name='rpn_rois',
                                  shape=[4, 4],
                                  dtype='float32',
                                  lod_level=1)
            gt_classes = fluid.data(name='gt_classes',
                                    shape=[6],
                                    dtype='int32',
                                    lod_level=1)
            is_crowd = fluid.data(name='is_crowd',
                                  shape=[6],
                                  dtype='int32',
                                  lod_level=1)
            gt_boxes = fluid.data(name='gt_boxes',
                                  shape=[6, 4],
                                  dtype='float32',
                                  lod_level=1)
            im_info = fluid.data(name='im_info', shape=[1, 3], dtype='float32')
            max_overlap = fluid.data(name='max_overlap',
                                     shape=[4],
                                     dtype='float32',
                                     lod_level=1)
            self.class_nums = 5
            outs = fluid.layers.generate_proposal_labels(
                rpn_rois=rpn_rois,
                gt_classes=gt_classes,
                is_crowd=is_crowd,
                gt_boxes=gt_boxes,
                im_info=im_info,
                batch_size_per_im=2,
                fg_fraction=0.5,
                fg_thresh=0.5,
                bg_thresh_hi=0.5,
                bg_thresh_lo=0.0,
                bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                class_nums=self.class_nums)
            outs_1 = fluid.layers.generate_proposal_labels(
                rpn_rois=rpn_rois,
                gt_classes=gt_classes,
                is_crowd=is_crowd,
                gt_boxes=gt_boxes,
                im_info=im_info,
                batch_size_per_im=2,
                fg_fraction=0.5,
                fg_thresh=0.5,
                bg_thresh_hi=0.5,
                bg_thresh_lo=0.0,
                bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
                class_nums=self.class_nums,
                is_cascade_rcnn=True,
                max_overlap=max_overlap,
                return_max_overlap=True)

            self.check_out(outs)
            self.check_out(outs_1)
            rois = outs[0]


class TestGenerateMaskLabels(unittest.TestCase):

    def test_generate_mask_labels(self):
        program = Program()
        with program_guard(program):
            im_info = layers.data(name='im_info',
                                  shape=[1, 3],
                                  dtype='float32',
                                  lod_level=1,
                                  append_batch_size=False)
            gt_classes = layers.data(name='gt_classes',
                                     shape=[2, 1],
                                     dtype='int32',
                                     lod_level=1,
                                     append_batch_size=False)
            is_crowd = layers.data(name='is_crowd',
                                   shape=[2, 1],
                                   dtype='int32',
                                   lod_level=1,
                                   append_batch_size=False)
            gt_segms = layers.data(name='gt_segms',
                                   shape=[20, 2],
                                   dtype='float32',
                                   lod_level=3,
                                   append_batch_size=False)
            rois = layers.data(name='rois',
                               shape=[4, 4],
                               dtype='float32',
                               lod_level=1,
                               append_batch_size=False)
            labels_int32 = layers.data(name='labels_int32',
                                       shape=[4, 1],
                                       dtype='int32',
                                       lod_level=1,
                                       append_batch_size=False)
            num_classes = 5
            resolution = 14
            outs = fluid.layers.generate_mask_labels(im_info=im_info,
                                                     gt_classes=gt_classes,
                                                     is_crowd=is_crowd,
                                                     gt_segms=gt_segms,
                                                     rois=rois,
                                                     labels_int32=labels_int32,
                                                     num_classes=num_classes,
                                                     resolution=resolution)
            mask_rois, roi_has_mask_int32, mask_int32 = outs
            assert mask_rois.shape[1] == 4
            assert mask_int32.shape[1] == num_classes * resolution * resolution


class TestMultiBoxHead(unittest.TestCase):

    def test_multi_box_head(self):
        data_shape = [3, 224, 224]
        mbox_locs, mbox_confs, box, var = self.multi_box_head_output(data_shape)

        assert len(box.shape) == 2
        assert box.shape == var.shape
        assert box.shape[1] == 4
        assert mbox_locs.shape[1] == mbox_confs.shape[1]

    def multi_box_head_output(self, data_shape):
        images = fluid.layers.data(name='pixel',
                                   shape=data_shape,
                                   dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        conv2 = fluid.layers.conv2d(conv1, 3, 3, 2)
        conv3 = fluid.layers.conv2d(conv2, 3, 3, 2)
        conv4 = fluid.layers.conv2d(conv3, 3, 3, 2)
        conv5 = fluid.layers.conv2d(conv4, 3, 3, 2)

        mbox_locs, mbox_confs, box, var = layers.multi_box_head(
            inputs=[conv1, conv2, conv3, conv4, conv5, conv5],
            image=images,
            num_classes=21,
            min_ratio=20,
            max_ratio=90,
            aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2.], [2.]],
            base_size=300,
            offset=0.5,
            flip=True,
            clip=True)

        return mbox_locs, mbox_confs, box, var


class TestDetectionMAP(unittest.TestCase):

    def test_detection_map(self):
        program = Program()
        with program_guard(program):
            detect_res = layers.data(name='detect_res',
                                     shape=[10, 6],
                                     append_batch_size=False,
                                     dtype='float32')
            label = layers.data(name='label',
                                shape=[10, 6],
                                append_batch_size=False,
                                dtype='float32')

            map_out = detection.detection_map(detect_res, label, 21)
            self.assertIsNotNone(map_out)
            self.assertEqual(map_out.shape, (1, ))
        print(str(program))


class TestRpnTargetAssign(unittest.TestCase):

    def test_rpn_target_assign(self):
        program = Program()
        with program_guard(program):
            bbox_pred_shape = [10, 50, 4]
            cls_logits_shape = [10, 50, 2]
            anchor_shape = [50, 4]

            bbox_pred = layers.data(name='bbox_pred',
                                    shape=bbox_pred_shape,
                                    append_batch_size=False,
                                    dtype='float32')
            cls_logits = layers.data(name='cls_logits',
                                     shape=cls_logits_shape,
                                     append_batch_size=False,
                                     dtype='float32')
            anchor_box = layers.data(name='anchor_box',
                                     shape=anchor_shape,
                                     append_batch_size=False,
                                     dtype='float32')
            anchor_var = layers.data(name='anchor_var',
                                     shape=anchor_shape,
                                     append_batch_size=False,
                                     dtype='float32')
            gt_boxes = layers.data(name='gt_boxes',
                                   shape=[4],
                                   lod_level=1,
                                   dtype='float32')
            is_crowd = layers.data(name='is_crowd',
                                   shape=[1, 10],
                                   dtype='int32',
                                   lod_level=1,
                                   append_batch_size=False)
            im_info = layers.data(name='im_info',
                                  shape=[1, 3],
                                  dtype='float32',
                                  lod_level=1,
                                  append_batch_size=False)
            outs = layers.rpn_target_assign(bbox_pred=bbox_pred,
                                            cls_logits=cls_logits,
                                            anchor_box=anchor_box,
                                            anchor_var=anchor_var,
                                            gt_boxes=gt_boxes,
                                            is_crowd=is_crowd,
                                            im_info=im_info,
                                            rpn_batch_size_per_im=256,
                                            rpn_straddle_thresh=0.0,
                                            rpn_fg_fraction=0.5,
                                            rpn_positive_overlap=0.7,
                                            rpn_negative_overlap=0.3,
                                            use_random=False)
            pred_scores = outs[0]
            pred_loc = outs[1]
            tgt_lbl = outs[2]
            tgt_bbox = outs[3]
            bbox_inside_weight = outs[4]

            self.assertIsNotNone(pred_scores)
            self.assertIsNotNone(pred_loc)
            self.assertIsNotNone(tgt_lbl)
            self.assertIsNotNone(tgt_bbox)
            self.assertIsNotNone(bbox_inside_weight)
            assert pred_scores.shape[1] == 1
            assert pred_loc.shape[1] == 4
            assert pred_loc.shape[1] == tgt_bbox.shape[1]
            print(str(program))


class TestGenerateProposals(LayerTest):

    def test_generate_proposals(self):
        scores_np = np.random.rand(2, 3, 4, 4).astype('float32')
        bbox_deltas_np = np.random.rand(2, 12, 4, 4).astype('float32')
        im_info_np = np.array([[8, 8, 0.5], [6, 6, 0.5]]).astype('float32')
        anchors_np = np.reshape(np.arange(4 * 4 * 3 * 4),
                                [4, 4, 3, 4]).astype('float32')
        variances_np = np.ones((4, 4, 3, 4)).astype('float32')

        with self.static_graph():
            scores = fluid.data(name='scores',
                                shape=[2, 3, 4, 4],
                                dtype='float32')
            bbox_deltas = fluid.data(name='bbox_deltas',
                                     shape=[2, 12, 4, 4],
                                     dtype='float32')
            im_info = fluid.data(name='im_info', shape=[2, 3], dtype='float32')
            anchors = fluid.data(name='anchors',
                                 shape=[4, 4, 3, 4],
                                 dtype='float32')
            variances = fluid.data(name='var',
                                   shape=[4, 4, 3, 4],
                                   dtype='float32')
            rois, roi_probs, rois_num = fluid.layers.generate_proposals(
                scores,
                bbox_deltas,
                im_info,
                anchors,
                variances,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True)
            rois_stat, roi_probs_stat, rois_num_stat = self.get_static_graph_result(
                feed={
                    'scores': scores_np,
                    'bbox_deltas': bbox_deltas_np,
                    'im_info': im_info_np,
                    'anchors': anchors_np,
                    'var': variances_np
                },
                fetch_list=[rois, roi_probs, rois_num],
                with_lod=False)

        with self.dynamic_graph():
            scores_dy = base.to_variable(scores_np)
            bbox_deltas_dy = base.to_variable(bbox_deltas_np)
            im_info_dy = base.to_variable(im_info_np)
            anchors_dy = base.to_variable(anchors_np)
            variances_dy = base.to_variable(variances_np)
            rois, roi_probs, rois_num = fluid.layers.generate_proposals(
                scores_dy,
                bbox_deltas_dy,
                im_info_dy,
                anchors_dy,
                variances_dy,
                pre_nms_top_n=10,
                post_nms_top_n=5,
                return_rois_num=True)
            rois_dy = rois.numpy()
            roi_probs_dy = roi_probs.numpy()
            rois_num_dy = rois_num.numpy()

        np.testing.assert_array_equal(np.array(rois_stat), rois_dy)
        np.testing.assert_array_equal(np.array(roi_probs_stat), roi_probs_dy)
        np.testing.assert_array_equal(np.array(rois_num_stat), rois_num_dy)


class TestYoloDetection(unittest.TestCase):

    def test_yolov3_loss(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            gt_box = layers.data(name='gt_box', shape=[10, 4], dtype='float32')
            gt_label = layers.data(name='gt_label', shape=[10], dtype='int32')
            gt_score = layers.data(name='gt_score', shape=[10], dtype='float32')
            loss = layers.yolov3_loss(x,
                                      gt_box,
                                      gt_label, [10, 13, 30, 13], [0, 1],
                                      10,
                                      0.7,
                                      32,
                                      gt_score=gt_score,
                                      use_label_smooth=False)

            self.assertIsNotNone(loss)

    def test_yolo_box(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            img_size = layers.data(name='img_size', shape=[2], dtype='int32')
            boxes, scores = layers.yolo_box(x, img_size, [10, 13, 30, 13], 10,
                                            0.01, 32)
            self.assertIsNotNone(boxes)
            self.assertIsNotNone(scores)

    def test_yolov3_loss_with_scale(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            gt_box = layers.data(name='gt_box', shape=[10, 4], dtype='float32')
            gt_label = layers.data(name='gt_label', shape=[10], dtype='int32')
            gt_score = layers.data(name='gt_score', shape=[10], dtype='float32')
            loss = layers.yolov3_loss(x,
                                      gt_box,
                                      gt_label, [10, 13, 30, 13], [0, 1],
                                      10,
                                      0.7,
                                      32,
                                      gt_score=gt_score,
                                      use_label_smooth=False,
                                      scale_x_y=1.2)

            self.assertIsNotNone(loss)

    def test_yolo_box_with_scale(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            img_size = layers.data(name='img_size', shape=[2], dtype='int32')
            boxes, scores = layers.yolo_box(x,
                                            img_size, [10, 13, 30, 13],
                                            10,
                                            0.01,
                                            32,
                                            scale_x_y=1.2)
            self.assertIsNotNone(boxes)
            self.assertIsNotNone(scores)


class TestBoxClip(unittest.TestCase):

    def test_box_clip(self):
        program = Program()
        with program_guard(program):
            input_box = layers.data(name='input_box',
                                    shape=[7, 4],
                                    dtype='float32',
                                    lod_level=1)
            im_info = layers.data(name='im_info', shape=[3], dtype='float32')
            out = layers.box_clip(input_box, im_info)
            self.assertIsNotNone(out)


class TestMulticlassNMS(unittest.TestCase):

    def test_multiclass_nms(self):
        program = Program()
        with program_guard(program):
            bboxes = layers.data(name='bboxes',
                                 shape=[-1, 10, 4],
                                 dtype='float32')
            scores = layers.data(name='scores', shape=[-1, 10], dtype='float32')
            output = layers.multiclass_nms(bboxes, scores, 0.3, 400, 200, 0.7)
            self.assertIsNotNone(output)

    def test_multiclass_nms_error(self):
        program = Program()
        with program_guard(program):
            bboxes1 = fluid.data(name='bboxes1',
                                 shape=[10, 10, 4],
                                 dtype='int32')
            scores1 = fluid.data(name='scores1',
                                 shape=[10, 10],
                                 dtype='float32')
            bboxes2 = fluid.data(name='bboxes2',
                                 shape=[10, 10, 4],
                                 dtype='float32')
            scores2 = fluid.data(name='scores2', shape=[10, 10], dtype='int32')
            self.assertRaises(TypeError,
                              layers.multiclass_nms,
                              bboxes=bboxes1,
                              scores=scores1,
                              score_threshold=0.5,
                              nms_top_k=400,
                              keep_top_k=200)
            self.assertRaises(TypeError,
                              layers.multiclass_nms,
                              bboxes=bboxes2,
                              scores=scores2,
                              score_threshold=0.5,
                              nms_top_k=400,
                              keep_top_k=200)


class TestMulticlassNMS2(unittest.TestCase):

    def test_multiclass_nms2(self):
        program = Program()
        with program_guard(program):
            bboxes = layers.data(name='bboxes',
                                 shape=[-1, 10, 4],
                                 dtype='float32')
            scores = layers.data(name='scores', shape=[-1, 10], dtype='float32')
            output = fluid.contrib.multiclass_nms2(bboxes, scores, 0.3, 400,
                                                   200, 0.7)
            output2, index = fluid.contrib.multiclass_nms2(bboxes,
                                                           scores,
                                                           0.3,
                                                           400,
                                                           200,
                                                           0.7,
                                                           return_index=True)
            self.assertIsNotNone(output)
            self.assertIsNotNone(output2)
            self.assertIsNotNone(index)


class TestCollectFpnPropsals(LayerTest):

    def test_collect_fpn_proposals(self):
        multi_bboxes_np = []
        multi_scores_np = []
        rois_num_per_level_np = []
        for i in range(4):
            bboxes_np = np.random.rand(5, 4).astype('float32')
            scores_np = np.random.rand(5, 1).astype('float32')
            rois_num = np.array([2, 3]).astype('int32')
            multi_bboxes_np.append(bboxes_np)
            multi_scores_np.append(scores_np)
            rois_num_per_level_np.append(rois_num)

        with self.static_graph():
            multi_bboxes = []
            multi_scores = []
            rois_num_per_level = []
            for i in range(4):
                bboxes = fluid.data(name='rois' + str(i),
                                    shape=[5, 4],
                                    dtype='float32',
                                    lod_level=1)
                scores = fluid.data(name='scores' + str(i),
                                    shape=[5, 1],
                                    dtype='float32',
                                    lod_level=1)
                rois_num = fluid.data(name='rois_num' + str(i),
                                      shape=[None],
                                      dtype='int32')

                multi_bboxes.append(bboxes)
                multi_scores.append(scores)
                rois_num_per_level.append(rois_num)

            fpn_rois, rois_num = layers.collect_fpn_proposals(
                multi_bboxes,
                multi_scores,
                2,
                5,
                10,
                rois_num_per_level=rois_num_per_level)
            feed = {}
            for i in range(4):
                feed['rois' + str(i)] = multi_bboxes_np[i]
                feed['scores' + str(i)] = multi_scores_np[i]
                feed['rois_num' + str(i)] = rois_num_per_level_np[i]
            fpn_rois_stat, rois_num_stat = self.get_static_graph_result(
                feed=feed, fetch_list=[fpn_rois, rois_num], with_lod=True)
            fpn_rois_stat = np.array(fpn_rois_stat)
            rois_num_stat = np.array(rois_num_stat)

        with self.dynamic_graph():
            multi_bboxes_dy = []
            multi_scores_dy = []
            rois_num_per_level_dy = []
            for i in range(4):
                bboxes_dy = base.to_variable(multi_bboxes_np[i])
                scores_dy = base.to_variable(multi_scores_np[i])
                rois_num_dy = base.to_variable(rois_num_per_level_np[i])
                multi_bboxes_dy.append(bboxes_dy)
                multi_scores_dy.append(scores_dy)
                rois_num_per_level_dy.append(rois_num_dy)
            fpn_rois_dy, rois_num_dy = fluid.layers.collect_fpn_proposals(
                multi_bboxes_dy,
                multi_scores_dy,
                2,
                5,
                10,
                rois_num_per_level=rois_num_per_level_dy)
            fpn_rois_dy = fpn_rois_dy.numpy()
            rois_num_dy = rois_num_dy.numpy()

        np.testing.assert_array_equal(fpn_rois_stat, fpn_rois_dy)
        np.testing.assert_array_equal(rois_num_stat, rois_num_dy)

    def test_collect_fpn_proposals_error(self):

        def generate_input(bbox_type, score_type, name):
            multi_bboxes = []
            multi_scores = []
            for i in range(4):
                bboxes = fluid.data(name='rois' + name + str(i),
                                    shape=[10, 4],
                                    dtype=bbox_type,
                                    lod_level=1)
                scores = fluid.data(name='scores' + name + str(i),
                                    shape=[10, 1],
                                    dtype=score_type,
                                    lod_level=1)
                multi_bboxes.append(bboxes)
                multi_scores.append(scores)
            return multi_bboxes, multi_scores

        program = Program()
        with program_guard(program):
            bbox1 = fluid.data(name='rois',
                               shape=[5, 10, 4],
                               dtype='float32',
                               lod_level=1)
            score1 = fluid.data(name='scores',
                                shape=[5, 10, 1],
                                dtype='float32',
                                lod_level=1)
            bbox2, score2 = generate_input('int32', 'float32', '2')
            self.assertRaises(TypeError,
                              layers.collect_fpn_proposals,
                              multi_rois=bbox1,
                              multi_scores=score1,
                              min_level=2,
                              max_level=5,
                              post_nms_top_n=2000)
            self.assertRaises(TypeError,
                              layers.collect_fpn_proposals,
                              multi_rois=bbox2,
                              multi_scores=score2,
                              min_level=2,
                              max_level=5,
                              post_nms_top_n=2000)


class TestDistributeFpnProposals(LayerTest):

    def test_distribute_fpn_proposals(self):
        rois_np = np.random.rand(10, 4).astype('float32')
        rois_num_np = np.array([4, 6]).astype('int32')
        with self.static_graph():
            rois = fluid.data(name='rois', shape=[10, 4], dtype='float32')
            rois_num = fluid.data(name='rois_num', shape=[None], dtype='int32')
            multi_rois, restore_ind, rois_num_per_level = layers.distribute_fpn_proposals(
                fpn_rois=rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num)
            fetch_list = multi_rois + [restore_ind] + rois_num_per_level
            output_stat = self.get_static_graph_result(feed={
                'rois': rois_np,
                'rois_num': rois_num_np
            },
                                                       fetch_list=fetch_list,
                                                       with_lod=True)
            output_stat_np = []
            for output in output_stat:
                output_np = np.array(output)
                if len(output_np) > 0:
                    output_stat_np.append(output_np)

        with self.dynamic_graph():
            rois_dy = base.to_variable(rois_np)
            rois_num_dy = base.to_variable(rois_num_np)
            multi_rois_dy, restore_ind_dy, rois_num_per_level_dy = layers.distribute_fpn_proposals(
                fpn_rois=rois_dy,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224,
                rois_num=rois_num_dy)
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
            fpn_rois = fluid.data(name='data_error',
                                  shape=[10, 4],
                                  dtype='int32',
                                  lod_level=1)
            self.assertRaises(TypeError,
                              layers.distribute_fpn_proposals,
                              fpn_rois=fpn_rois,
                              min_level=2,
                              max_level=5,
                              refer_level=4,
                              refer_scale=224)


class TestBoxDecoderAndAssign(unittest.TestCase):

    def test_box_decoder_and_assign(self):
        program = Program()
        with program_guard(program):
            pb = fluid.data(name='prior_box', shape=[None, 4], dtype='float32')
            pbv = fluid.data(name='prior_box_var', shape=[4], dtype='float32')
            loc = fluid.data(name='target_box',
                             shape=[None, 4 * 81],
                             dtype='float32')
            scores = fluid.data(name='scores',
                                shape=[None, 81],
                                dtype='float32')
            decoded_box, output_assign_box = fluid.layers.box_decoder_and_assign(
                pb, pbv, loc, scores, 4.135)
            self.assertIsNotNone(decoded_box)
            self.assertIsNotNone(output_assign_box)

    def test_box_decoder_and_assign_error(self):

        def generate_input(pb_type, pbv_type, loc_type, score_type, name):
            pb = fluid.data(name='prior_box' + name,
                            shape=[None, 4],
                            dtype=pb_type)
            pbv = fluid.data(name='prior_box_var' + name,
                             shape=[4],
                             dtype=pbv_type)
            loc = fluid.data(name='target_box' + name,
                             shape=[None, 4 * 81],
                             dtype=loc_type)
            scores = fluid.data(name='scores' + name,
                                shape=[None, 81],
                                dtype=score_type)
            return pb, pbv, loc, scores

        program = Program()
        with program_guard(program):
            pb1, pbv1, loc1, scores1 = generate_input('int32', 'float32',
                                                      'float32', 'float32', '1')
            pb2, pbv2, loc2, scores2 = generate_input('float32', 'float32',
                                                      'int32', 'float32', '2')
            pb3, pbv3, loc3, scores3 = generate_input('float32', 'float32',
                                                      'float32', 'int32', '3')
            self.assertRaises(TypeError,
                              layers.box_decoder_and_assign,
                              prior_box=pb1,
                              prior_box_var=pbv1,
                              target_box=loc1,
                              box_score=scores1,
                              box_clip=4.0)
            self.assertRaises(TypeError,
                              layers.box_decoder_and_assign,
                              prior_box=pb2,
                              prior_box_var=pbv2,
                              target_box=loc2,
                              box_score=scores2,
                              box_clip=4.0)
            self.assertRaises(TypeError,
                              layers.box_decoder_and_assign,
                              prior_box=pb3,
                              prior_box_var=pbv3,
                              target_box=loc3,
                              box_score=scores3,
                              box_clip=4.0)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
