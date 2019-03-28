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

    def test_box_coder_api(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[4], dtype='float32')
            y = layers.data(name='z', shape=[4], dtype='float32', lod_level=1)
            bcoder = layers.box_coder(
                prior_box=x,
                prior_box_var=[0.1, 0.2, 0.1, 0.2],
                target_box=y,
                code_type='encode_center_size')
            self.assertIsNotNone(bcoder)
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
        program = Program()
        with program_guard(program):
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


class TestDensityPriorBox(unittest.TestCase):
    def test_density_prior_box(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 224, 224]
            images = fluid.layers.data(
                name='pixel', shape=data_shape, dtype='float32')
            conv1 = fluid.layers.conv2d(images, 3, 3, 2)
            box, var = layers.density_prior_box(
                input=conv1,
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
        program = Program()
        with program_guard(program):
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
            is_crowd = layers.data(
                name='is_crowd',
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
            im_info = layers.data(
                name='im_info',
                shape=[1, 3],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            class_nums = 5
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
                class_nums=class_nums)
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
            assert bbox_targets.shape[1] == 4 * class_nums
            assert bbox_inside_weights.shape[1] == 4 * class_nums
            assert bbox_outside_weights.shape[1] == 4 * class_nums


class TestGenerateMaskLabels(unittest.TestCase):
    def test_generate_mask_labels(self):
        program = Program()
        with program_guard(program):
            im_info = layers.data(
                name='im_info',
                shape=[1, 3],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            gt_classes = layers.data(
                name='gt_classes',
                shape=[2, 1],
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            is_crowd = layers.data(
                name='is_crowd',
                shape=[2, 1],
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            gt_segms = layers.data(
                name='gt_segms',
                shape=[20, 2],
                dtype='float32',
                lod_level=3,
                append_batch_size=False)
            rois = layers.data(
                name='rois',
                shape=[4, 4],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            labels_int32 = layers.data(
                name='labels_int32',
                shape=[4, 1],
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            num_classes = 5
            resolution = 14
            outs = fluid.layers.generate_mask_labels(
                im_info=im_info,
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
            bbox_pred_shape = [10, 50, 4]
            cls_logits_shape = [10, 50, 2]
            anchor_shape = [50, 4]

            bbox_pred = layers.data(
                name='bbox_pred',
                shape=bbox_pred_shape,
                append_batch_size=False,
                dtype='float32')
            cls_logits = layers.data(
                name='cls_logits',
                shape=cls_logits_shape,
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
            gt_boxes = layers.data(
                name='gt_boxes', shape=[4], lod_level=1, dtype='float32')
            is_crowd = layers.data(
                name='is_crowd',
                shape=[1, 10],
                dtype='int32',
                lod_level=1,
                append_batch_size=False)
            im_info = layers.data(
                name='im_info',
                shape=[1, 3],
                dtype='float32',
                lod_level=1,
                append_batch_size=False)
            outs = layers.rpn_target_assign(
                bbox_pred=bbox_pred,
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


class TestGenerateProposals(unittest.TestCase):
    def test_generate_proposals(self):
        program = Program()
        with program_guard(program):
            data_shape = [20, 64, 64]
            images = fluid.layers.data(
                name='images', shape=data_shape, dtype='float32')
            im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32')
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
                name='scores', shape=[num_anchors, 8, 8], dtype='float32')
            bbox_deltas = fluid.layers.data(
                name='bbox_deltas',
                shape=[num_anchors * 4, 8, 8],
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


class TestYoloDetection(unittest.TestCase):
    def test_yolov3_loss(self):
        program = Program()
        with program_guard(program):
            x = layers.data(name='x', shape=[30, 7, 7], dtype='float32')
            gtbox = layers.data(name='gtbox', shape=[10, 4], dtype='float32')
            gtlabel = layers.data(name='gtlabel', shape=[10], dtype='int32')
            gtscore = layers.data(name='gtscore', shape=[10], dtype='float32')
            loss = layers.yolov3_loss(
                x,
                gtbox,
                gtlabel, [10, 13, 30, 13], [0, 1],
                10,
                0.7,
                32,
                gtscore=gtscore,
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


class TestBoxClip(unittest.TestCase):
    def test_box_clip(self):
        program = Program()
        with program_guard(program):
            input_box = layers.data(
                name='input_box', shape=[7, 4], dtype='float32', lod_level=1)
            im_info = layers.data(name='im_info', shape=[3], dtype='float32')
            out = layers.box_clip(input_box, im_info)
            self.assertIsNotNone(out)


class TestMulticlassNMS(unittest.TestCase):
    def test_multiclass_nms(self):
        program = Program()
        with program_guard(program):
            bboxes = layers.data(
                name='bboxes', shape=[-1, 10, 4], dtype='float32')
            scores = layers.data(name='scores', shape=[-1, 10], dtype='float32')
            output = layers.multiclass_nms(bboxes, scores, 0.3, 400, 200, 0.7)
            self.assertIsNotNone(output)


class TestDistributeFpnProposals(unittest.TestCase):
    def test_distribute_fpn_proposals(self):
        program = Program()
        with program_guard(program):
            fpn_rois = fluid.layers.data(
                name='data', shape=[4], dtype='float32', lod_level=1)
            multi_rois, restore_ind = layers.distribute_fpn_proposals(
                fpn_rois=fpn_rois,
                min_level=2,
                max_level=5,
                refer_level=4,
                refer_scale=224)
            self.assertIsNotNone(multi_rois)
            self.assertIsNotNone(restore_ind)


if __name__ == '__main__':
    unittest.main()
