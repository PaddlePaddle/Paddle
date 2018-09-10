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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.framework import Program, program_guard
import unittest


class TestDetection(unittest.TestCase):
    def test_detection_output(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(
                name='target_box',
                shape=[2, 10, 4],
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=[2, 10, 20],
                append_batch_size=False,
                dtype='float32')
            out = layers.detection_output(
                scores=scores, loc=loc, prior_box=pb, prior_box_var=pbv)
            self.assertIsNotNone(out)
            self.assertEqual(out.shape[-1], 6)
        print(str(program))

    def test_detection_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='y', shape=[4], dtype='float32')
            z = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            iou = layers.iou_similarity(x=x, y=y)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=y,
                target_box=z,
                code_type='encode_center_size')
            self.assertIsNotNone(iou)
            self.assertIsNotNone(bcoder)

            matched_indices, matched_dist = layers.bipartite_match(iou)
            self.assertIsNotNone(matched_indices)
            self.assertIsNotNone(matched_dist)

            gt = layers.data(
                name='gt', shape=[1, 1], dtype='int32', lod_level=1)
            trg, trg_weight = layers.target_assign(
                gt, matched_indices, mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

            gt2 = layers.data(
                name='gt2', shape=[10, 4], dtype='float32', lod_level=1)
            trg, trg_weight = layers.target_assign(
                gt2, matched_indices, mismatch_value=0)
            self.assertIsNotNone(trg)
            self.assertIsNotNone(trg_weight)

        print(str(program))

    def test_ssd_loss(self):
        program = Program()
        with program_guard(program):
            pb = layers.data(
                name='prior_box',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            pbv = layers.data(
                name='prior_box_var',
                shape=[10, 4],
                append_batch_size=False,
                dtype='float32')
            loc = layers.data(name='target_box', shape=[10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[10, 21], dtype='float32')
            gt_box = layers.data(
                name='gt_box', shape=[4], lod_level=1, dtype='float32')
            gt_label = layers.data(
                name='gt_label', shape=[1], lod_level=1, dtype='int32')
            loss = layers.ssd_loss(loc, scores, gt_box, gt_label, pb, pbv)
            self.assertIsNotNone(loss)
            self.assertEqual(loss.shape[-1], 1)
        print(str(program))


class TestPriorBox(unittest.TestCase):
    def test_prior_box(self):
        data_shape = [3, 224, 224]
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
        conv1 = fluid.layers.conv2d(images, 3, 3, 2)
        box, var = layers.prior_box(
            input=conv1,
            image=images,
            min_sizes=[100.0],
            aspect_ratios=[1.],
            flip=True,
            clip=True)
        assert len(box.shape) == 4
        assert box.shape == var.shape
        assert box.shape[3] == 4


class TestAnchorGenerator(unittest.TestCase):
    def test_anchor_generator(self):
        data_shape = [3, 224, 224]
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
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
    def test_generate_proposal_labels(self):
        rpn_rois = layers.data(
            name='rpn_rois',
            shape=[4, 4],
            dtype='float32',
            lod_level=1,
            append_batch_size=False)
        gt_classes = layers.data(
            name='gt_classes',
            shape=[6],
            dtype='int32',
            lod_level=1,
            append_batch_size=False)
        gt_boxes = layers.data(
            name='gt_boxes',
            shape=[6, 4],
            dtype='float32',
            lod_level=1,
            append_batch_size=False)
        im_scales = layers.data(
            name='im_scales',
            shape=[1],
            dtype='float32',
            lod_level=1,
            append_batch_size=False)
        class_nums = 5
        rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights = fluid.layers.generate_proposal_labels(
            rpn_rois=rpn_rois,
            gt_classes=gt_classes,
            gt_boxes=gt_boxes,
            im_scales=im_scales,
            batch_size_per_im=2,
            fg_fraction=0.5,
            fg_thresh=0.5,
            bg_thresh_hi=0.5,
            bg_thresh_lo=0.0,
            bbox_reg_weights=[0.1, 0.1, 0.2, 0.2],
            class_nums=class_nums)
        assert rois.shape[1] == 4
        assert rois.shape[0] == labels_int32.shape[0]
        assert rois.shape[0] == bbox_targets.shape[0]
        assert rois.shape[0] == bbox_inside_weights.shape[0]
        assert rois.shape[0] == bbox_outside_weights.shape[0]
        assert bbox_targets.shape[1] == 4 * class_nums
        assert bbox_inside_weights.shape[1] == 4 * class_nums
        assert bbox_outside_weights.shape[1] == 4 * class_nums


class TestMultiBoxHead(unittest.TestCase):
    def test_multi_box_head(self):
        data_shape = [3, 224, 224]
        mbox_locs, mbox_confs, box, var = self.multi_box_head_output(data_shape)

        assert len(box.shape) == 2
        assert box.shape == var.shape
        assert box.shape[1] == 4
        assert mbox_locs.shape[1] == mbox_confs.shape[1]

    def multi_box_head_output(self, data_shape):
        images = fluid.layers.data(
            name='pixel', shape=data_shape, dtype='float32')
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
            detect_res = layers.data(
                name='detect_res',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')
            label = layers.data(
                name='label',
                shape=[10, 6],
                append_batch_size=False,
                dtype='float32')

            map_out = layers.detection_map(detect_res, label, 21)
            self.assertIsNotNone(map_out)
            self.assertEqual(map_out.shape, (1, ))
        print(str(program))


class TestRpnTargetAssign(unittest.TestCase):
    def test_rpn_target_assign(self):
        program = Program()
        with program_guard(program):
            loc_shape = [10, 50, 4]
            score_shape = [10, 50, 2]
            anchor_shape = [50, 4]

            loc = layers.data(
                name='loc',
                shape=loc_shape,
                append_batch_size=False,
                dtype='float32')
            scores = layers.data(
                name='scores',
                shape=score_shape,
                append_batch_size=False,
                dtype='float32')
            anchor_box = layers.data(
                name='anchor_box',
                shape=anchor_shape,
                append_batch_size=False,
                dtype='float32')
            anchor_var = layers.data(
                name='anchor_var',
                shape=anchor_shape,
                append_batch_size=False,
                dtype='float32')
            gt_box = layers.data(
                name='gt_box', shape=[4], lod_level=1, dtype='float32')

            pred_scores, pred_loc, tgt_lbl, tgt_bbox = layers.rpn_target_assign(
                loc=loc,
                scores=scores,
                anchor_box=anchor_box,
                anchor_var=anchor_var,
                gt_box=gt_box,
                rpn_batch_size_per_im=256,
                fg_fraction=0.25,
                rpn_positive_overlap=0.7,
                rpn_negative_overlap=0.3)

            self.assertIsNotNone(pred_scores)
            self.assertIsNotNone(pred_loc)
            self.assertIsNotNone(tgt_lbl)
            self.assertIsNotNone(tgt_bbox)
            assert pred_scores.shape[1] == 1
            assert pred_loc.shape[1] == 4
            assert pred_loc.shape[1] == tgt_bbox.shape[1]


class TestGenerateProposals(unittest.TestCase):
    def test_generate_proposals(self):
        data_shape = [20, 64, 64]
        images = fluid.layers.data(
            name='images', shape=data_shape, dtype='float32')
        im_info = fluid.layers.data(
            name='im_info', shape=[1, 3], dtype='float32')
        anchors, variances = fluid.layers.anchor_generator(
            name='anchor_generator',
            input=images,
            anchor_sizes=[32, 64],
            aspect_ratios=[1.0],
            variance=[0.1, 0.1, 0.2, 0.2],
            stride=[16.0, 16.0],
            offset=0.5)
        num_anchors = anchors.shape[2]
        scores = fluid.layers.data(
            name='scores', shape=[1, num_anchors, 8, 8], dtype='float32')
        bbox_deltas = fluid.layers.data(
            name='bbox_deltas',
            shape=[1, num_anchors * 4, 8, 8],
            dtype='float32')
        rpn_rois, rpn_roi_probs = fluid.layers.generate_proposals(
            name='generate_proposals',
            scores=scores,
            bbox_deltas=bbox_deltas,
            im_info=im_info,
            anchors=anchors,
            variances=variances,
            pre_nms_top_n=6000,
            post_nms_top_n=1000,
            nms_thresh=0.5,
            min_size=0.1,
            eta=1.0)
        self.assertIsNotNone(rpn_rois)
        self.assertIsNotNone(rpn_roi_probs)
        print(rpn_rois.shape)


if __name__ == '__main__':
    unittest.main()
