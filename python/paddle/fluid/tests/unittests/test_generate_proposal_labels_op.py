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

import unittest
import numpy as np
import sys
import math
import paddle.fluid as fluid
from op_test import OpTest


def generate_proposal_labels_in_python(rpn_rois,
                                       gt_classes,
                                       is_crowd,
                                       gt_boxes,
                                       im_info,
                                       batch_size_per_im,
                                       fg_fraction,
                                       fg_thresh,
                                       bg_thresh_hi,
                                       bg_thresh_lo,
                                       bbox_reg_weights,
                                       class_nums,
                                       use_random,
                                       is_cls_agnostic,
                                       is_cascade_rcnn,
                                       max_overlaps=None):
    rois = []
    labels_int32 = []
    bbox_targets = []
    bbox_inside_weights = []
    bbox_outside_weights = []
    max_overlap_with_gt = []
    lod = []
    assert len(rpn_rois) == len(
        im_info), 'batch size of rpn_rois and ground_truth is not matched'

    for im_i in range(len(im_info)):
        max_overlap = max_overlaps[im_i] if is_cascade_rcnn else None
        frcn_blobs = _sample_rois(rpn_rois[im_i], gt_classes[im_i],
                                  is_crowd[im_i], gt_boxes[im_i], im_info[im_i],
                                  batch_size_per_im, fg_fraction, fg_thresh,
                                  bg_thresh_hi, bg_thresh_lo, bbox_reg_weights,
                                  class_nums, use_random, is_cls_agnostic,
                                  is_cascade_rcnn, max_overlap)
        lod.append(frcn_blobs['rois'].shape[0])
        rois.append(frcn_blobs['rois'])
        labels_int32.append(frcn_blobs['labels_int32'])
        bbox_targets.append(frcn_blobs['bbox_targets'])
        bbox_inside_weights.append(frcn_blobs['bbox_inside_weights'])
        bbox_outside_weights.append(frcn_blobs['bbox_outside_weights'])
        max_overlap_with_gt.append(frcn_blobs['max_overlap'])

    return rois, labels_int32, bbox_targets, bbox_inside_weights, bbox_outside_weights, max_overlap_with_gt, lod


def filter_roi(rois, max_overlap):
    ws = rois[:, 2] - rois[:, 0] + 1
    hs = rois[:, 3] - rois[:, 1] + 1
    keep = np.where((ws > 0) & (hs > 0) & (max_overlap < 1.0))[0]
    if len(keep) > 0:
        return rois[keep, :]
    return np.zeros((1, 4)).astype('float32')


def _sample_rois(rpn_rois, gt_classes, is_crowd, gt_boxes, im_info,
                 batch_size_per_im, fg_fraction, fg_thresh, bg_thresh_hi,
                 bg_thresh_lo, bbox_reg_weights, class_nums, use_random,
                 is_cls_agnostic, is_cascade_rcnn, max_overlap):
    rois_per_image = int(batch_size_per_im)
    fg_rois_per_im = int(np.round(fg_fraction * rois_per_image))

    # Roidb
    im_scale = im_info[2]
    inv_im_scale = 1. / im_scale
    rpn_rois = rpn_rois * inv_im_scale

    if is_cascade_rcnn:
        rpn_rois = filter_roi(rpn_rois, max_overlap)

    boxes = np.vstack([gt_boxes, rpn_rois])

    gt_overlaps = np.zeros((boxes.shape[0], class_nums))
    box_to_gt_ind_map = np.zeros((boxes.shape[0]), dtype=np.int32)
    proposal_to_gt_overlaps = _bbox_overlaps(boxes, gt_boxes)

    overlaps_argmax = proposal_to_gt_overlaps.argmax(axis=1)
    overlaps_max = proposal_to_gt_overlaps.max(axis=1)
    # Boxes which with non-zero overlap with gt boxes
    overlapped_boxes_ind = np.where(overlaps_max > 0)[0]
    overlapped_boxes_gt_classes = gt_classes[
        overlaps_argmax[overlapped_boxes_ind]]
    gt_overlaps[
        overlapped_boxes_ind,
        overlapped_boxes_gt_classes] = overlaps_max[overlapped_boxes_ind]
    box_to_gt_ind_map[overlapped_boxes_ind] = overlaps_argmax[
        overlapped_boxes_ind]

    crowd_ind = np.where(is_crowd)[0]
    gt_overlaps[crowd_ind] = -1.0
    max_overlaps = gt_overlaps.max(axis=1)
    max_classes = gt_overlaps.argmax(axis=1)

    if is_cascade_rcnn:
        # Cascade RCNN Decode Filter
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        bg_inds = np.where((max_overlaps < bg_thresh_hi)
                           & (max_overlaps >= bg_thresh_lo))[0]
        fg_rois_per_this_image = fg_inds.shape[0]
        bg_rois_per_this_image = bg_inds.shape[0]
    else:
        # Foreground
        fg_inds = np.where(max_overlaps >= fg_thresh)[0]
        fg_rois_per_this_image = np.minimum(fg_rois_per_im, fg_inds.shape[0])
        # Sample foreground if there are too many
        if (fg_inds.shape[0] > fg_rois_per_this_image) and use_random:
            fg_inds = np.random.choice(fg_inds,
                                       size=fg_rois_per_this_image,
                                       replace=False)
        fg_inds = fg_inds[:fg_rois_per_this_image]
        # Background
        bg_inds = np.where((max_overlaps < bg_thresh_hi)
                           & (max_overlaps >= bg_thresh_lo))[0]
        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
        bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                            bg_inds.shape[0])
        # Sample background if there are too many
        if (bg_inds.shape[0] > bg_rois_per_this_image) and use_random:
            bg_inds = np.random.choice(bg_inds,
                                       size=bg_rois_per_this_image,
                                       replace=False)
        bg_inds = bg_inds[:bg_rois_per_this_image]

    keep_inds = np.append(fg_inds, bg_inds)
    sampled_labels = max_classes[keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0
    sampled_boxes = boxes[keep_inds]
    sampled_max_overlap = max_overlaps[keep_inds]
    sampled_gts = gt_boxes[box_to_gt_ind_map[keep_inds]]
    sampled_gts[fg_rois_per_this_image:, :] = gt_boxes[0]
    bbox_label_targets = _compute_targets(sampled_boxes, sampled_gts,
                                          sampled_labels, bbox_reg_weights)
    bbox_targets, bbox_inside_weights = _expand_bbox_targets(
        bbox_label_targets, class_nums, is_cls_agnostic)
    bbox_outside_weights = np.array(bbox_inside_weights > 0,
                                    dtype=bbox_inside_weights.dtype)
    # Scale rois
    sampled_rois = sampled_boxes * im_scale

    # Faster RCNN blobs
    frcn_blobs = dict(rois=sampled_rois,
                      labels_int32=sampled_labels,
                      bbox_targets=bbox_targets,
                      bbox_inside_weights=bbox_inside_weights,
                      bbox_outside_weights=bbox_outside_weights,
                      max_overlap=sampled_max_overlap)
    return frcn_blobs


def _bbox_overlaps(roi_boxes, gt_boxes):
    w1 = np.maximum(roi_boxes[:, 2] - roi_boxes[:, 0] + 1, 0)
    h1 = np.maximum(roi_boxes[:, 3] - roi_boxes[:, 1] + 1, 0)
    w2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0] + 1, 0)
    h2 = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1] + 1, 0)
    area1 = w1 * h1
    area2 = w2 * h2

    overlaps = np.zeros((roi_boxes.shape[0], gt_boxes.shape[0]))
    for ind1 in range(roi_boxes.shape[0]):
        for ind2 in range(gt_boxes.shape[0]):
            inter_x1 = np.maximum(roi_boxes[ind1, 0], gt_boxes[ind2, 0])
            inter_y1 = np.maximum(roi_boxes[ind1, 1], gt_boxes[ind2, 1])
            inter_x2 = np.minimum(roi_boxes[ind1, 2], gt_boxes[ind2, 2])
            inter_y2 = np.minimum(roi_boxes[ind1, 3], gt_boxes[ind2, 3])
            inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
            inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
            inter_area = inter_w * inter_h
            iou = inter_area / (area1[ind1] + area2[ind2] - inter_area)
            overlaps[ind1, ind2] = iou
    return overlaps


def _compute_targets(roi_boxes, gt_boxes, labels, bbox_reg_weights):
    assert roi_boxes.shape[0] == gt_boxes.shape[0]
    assert roi_boxes.shape[1] == 4
    assert gt_boxes.shape[1] == 4

    targets = np.zeros(roi_boxes.shape)
    bbox_reg_weights = np.asarray(bbox_reg_weights)
    targets = _box_to_delta(ex_boxes=roi_boxes,
                            gt_boxes=gt_boxes,
                            weights=bbox_reg_weights)

    return np.hstack([labels[:, np.newaxis], targets]).astype(np.float32,
                                                              copy=False)


def _box_to_delta(ex_boxes, gt_boxes, weights):
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets


def _expand_bbox_targets(bbox_targets_input, class_nums, is_cls_agnostic):
    class_labels = bbox_targets_input[:, 0]
    fg_inds = np.where(class_labels > 0)[0]
    # if is_cls_agnostic:
    #     class_labels = [1 if ll > 0 else 0 for ll in class_labels]
    #     class_labels = np.array(class_labels, dtype=np.int32)
    #     class_nums = 2
    bbox_targets = np.zeros((class_labels.shape[0],
                             4 * class_nums if not is_cls_agnostic else 4 * 2))
    bbox_inside_weights = np.zeros(bbox_targets.shape)
    for ind in fg_inds:
        class_label = int(class_labels[ind]) if not is_cls_agnostic else 1
        start_ind = class_label * 4
        end_ind = class_label * 4 + 4
        bbox_targets[ind, start_ind:end_ind] = bbox_targets_input[ind, 1:]
        bbox_inside_weights[ind, start_ind:end_ind] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


class TestGenerateProposalLabelsOp(OpTest):

    def set_data(self):
        #self.use_random = False
        self.init_use_random()
        self.init_test_params()
        self.init_test_input()
        self.init_test_cascade()
        self.init_test_output()

        self.inputs = {
            'RpnRois': (self.rpn_rois[0], self.rpn_rois_lod),
            'GtClasses': (self.gt_classes[0], self.gts_lod),
            'IsCrowd': (self.is_crowd[0], self.gts_lod),
            'GtBoxes': (self.gt_boxes[0], self.gts_lod),
            'ImInfo': self.im_info,
        }
        if self.max_overlaps is not None:
            self.inputs['MaxOverlap'] = (self.max_overlaps[0],
                                         self.rpn_rois_lod)

        self.attrs = {
            'batch_size_per_im': self.batch_size_per_im,
            'fg_fraction': self.fg_fraction,
            'fg_thresh': self.fg_thresh,
            'bg_thresh_hi': self.bg_thresh_hi,
            'bg_thresh_lo': self.bg_thresh_lo,
            'bbox_reg_weights': self.bbox_reg_weights,
            'class_nums': self.class_nums,
            'use_random': self.use_random,
            'is_cls_agnostic': self.is_cls_agnostic,
            'is_cascade_rcnn': self.is_cascade_rcnn
        }
        self.outputs = {
            'Rois': (self.rois, [self.lod]),
            'LabelsInt32': (self.labels_int32, [self.lod]),
            'BboxTargets': (self.bbox_targets, [self.lod]),
            'BboxInsideWeights': (self.bbox_inside_weights, [self.lod]),
            'BboxOutsideWeights': (self.bbox_outside_weights, [self.lod]),
            'MaxOverlapWithGT': (self.max_overlap_with_gt, [self.lod]),
        }

    def test_check_output(self):
        self.check_output()

    def setUp(self):
        self.op_type = 'generate_proposal_labels'
        self.set_data()

    def init_test_cascade(self, ):
        self.is_cascade_rcnn = False
        self.max_overlaps = None

    def init_use_random(self):
        self.use_random = False

    def init_test_params(self):
        self.batch_size_per_im = 100
        self.fg_fraction = 0.25
        self.fg_thresh = 0.5
        self.bg_thresh_hi = 0.5
        self.bg_thresh_lo = 0.0
        self.bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]
        self.is_cls_agnostic = False
        self.class_nums = 2 if self.is_cls_agnostic else 81

    def init_test_input(self):
        np.random.seed(0)
        gt_nums = 6  # Keep same with batch_size_per_im for unittest
        proposal_nums = 200
        images_shape = [[64, 64]]
        self.im_info = np.ones((len(images_shape), 3)).astype(np.float32)
        for i in range(len(images_shape)):
            self.im_info[i, 0] = images_shape[i][0]
            self.im_info[i, 1] = images_shape[i][1]
            self.im_info[i, 2] = 0.8  #scale

        self.rpn_rois, self.rpn_rois_lod = _generate_proposals(
            images_shape, proposal_nums)
        ground_truth, self.gts_lod = _generate_groundtruth(
            images_shape, self.class_nums, gt_nums)

        self.gt_classes = [gt['gt_classes'] for gt in ground_truth]
        self.gt_boxes = [gt['boxes'] for gt in ground_truth]
        self.is_crowd = [gt['is_crowd'] for gt in ground_truth]

    def init_test_output(self):
        self.rois, self.labels_int32, self.bbox_targets, \
        self.bbox_inside_weights, self.bbox_outside_weights, \
        self.max_overlap_with_gt, \
        self.lod = generate_proposal_labels_in_python(
                self.rpn_rois, self.gt_classes, self.is_crowd, self.gt_boxes, self.im_info,
                self.batch_size_per_im, self.fg_fraction,
                self.fg_thresh, self.bg_thresh_hi, self.bg_thresh_lo,
                self.bbox_reg_weights, self.class_nums, self.use_random,
                self.is_cls_agnostic, self.is_cascade_rcnn, self.max_overlaps
            )
        self.rois = np.vstack(self.rois)
        self.labels_int32 = np.hstack(self.labels_int32)
        self.labels_int32 = self.labels_int32[:, np.newaxis]
        self.bbox_targets = np.vstack(self.bbox_targets)
        self.bbox_inside_weights = np.vstack(self.bbox_inside_weights)
        self.bbox_outside_weights = np.vstack(self.bbox_outside_weights)
        self.max_overlap_with_gt = np.vstack(self.max_overlap_with_gt)


class TestCascade(TestGenerateProposalLabelsOp):

    def init_test_cascade(self):
        self.is_cascade_rcnn = True
        roi_num = len(self.rpn_rois[0])
        self.max_overlaps = []
        max_overlap = np.random.rand(roi_num).astype('float32')
        # Make GT samples with overlap = 1
        max_overlap[max_overlap > 0.9] = 1.
        self.max_overlaps.append(max_overlap)


class TestUseRandom(TestGenerateProposalLabelsOp):

    def init_use_random(self):
        self.use_random = True
        self.is_cascade_rcnn = False

    def test_check_output(self):
        self.check_output_customized(self.verify_out)

    def verify_out(self, outs):
        print("skip")

    def init_test_params(self):
        self.batch_size_per_im = 512
        self.fg_fraction = 0.025
        self.fg_thresh = 0.5
        self.bg_thresh_hi = 0.5
        self.bg_thresh_lo = 0.0
        self.bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]
        self.is_cls_agnostic = False
        self.class_nums = 2 if self.is_cls_agnostic else 81


class TestClsAgnostic(TestCascade):

    def init_test_params(self):
        self.batch_size_per_im = 512
        self.fg_fraction = 0.25
        self.fg_thresh = 0.5
        self.bg_thresh_hi = 0.5
        self.bg_thresh_lo = 0.0
        self.bbox_reg_weights = [0.1, 0.1, 0.2, 0.2]
        self.is_cls_agnostic = True
        self.class_nums = 2 if self.is_cls_agnostic else 81


class TestOnlyGT(TestCascade):

    def init_test_input(self):
        np.random.seed(0)
        gt_nums = 6  # Keep same with batch_size_per_im for unittest
        proposal_nums = 6
        images_shape = [[64, 64]]
        self.im_info = np.ones((len(images_shape), 3)).astype(np.float32)
        for i in range(len(images_shape)):
            self.im_info[i, 0] = images_shape[i][0]
            self.im_info[i, 1] = images_shape[i][1]
            self.im_info[i, 2] = 0.8  #scale

        ground_truth, self.gts_lod = _generate_groundtruth(
            images_shape, self.class_nums, gt_nums)

        self.gt_classes = [gt['gt_classes'] for gt in ground_truth]
        self.gt_boxes = [gt['boxes'] for gt in ground_truth]
        self.is_crowd = [gt['is_crowd'] for gt in ground_truth]
        self.rpn_rois = self.gt_boxes
        self.rpn_rois_lod = self.gts_lod


class TestOnlyGT2(TestCascade):

    def init_test_cascade(self):
        self.is_cascade_rcnn = True
        roi_num = len(self.rpn_rois[0])
        self.max_overlaps = []
        max_overlap = np.ones(roi_num).astype('float32')
        self.max_overlaps.append(max_overlap)


def _generate_proposals(images_shape, proposal_nums):
    rpn_rois = []
    rpn_rois_lod = []
    num_proposals = 0
    for i, image_shape in enumerate(images_shape):
        proposals = _generate_boxes(image_shape, proposal_nums)
        rpn_rois.append(proposals)
        num_proposals = len(proposals)
        rpn_rois_lod.append(num_proposals)
    return rpn_rois, [rpn_rois_lod]


def _generate_groundtruth(images_shape, class_nums, gt_nums):
    ground_truth = []
    gts_lod = []
    num_gts = 0
    for i, image_shape in enumerate(images_shape):
        # Avoid background
        gt_classes = np.random.randint(low=1, high=class_nums,
                                       size=gt_nums).astype(np.int32)
        gt_boxes = _generate_boxes(image_shape, gt_nums)
        is_crowd = np.zeros((gt_nums), dtype=np.int32)
        is_crowd[0] = 1
        ground_truth.append(
            dict(gt_classes=gt_classes, boxes=gt_boxes, is_crowd=is_crowd))
        num_gts += len(gt_classes)
        gts_lod.append(num_gts)
    return ground_truth, [gts_lod]


def _generate_boxes(image_size, box_nums):
    width = image_size[0]
    height = image_size[1]
    xywh = np.random.rand(box_nums, 4)
    xy1 = xywh[:, [0, 1]] * image_size
    wh = xywh[:, [2, 3]] * (image_size - xy1)
    xy2 = xy1 + wh
    boxes = np.hstack([xy1, xy2])
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes.astype(np.float32)


if __name__ == '__main__':
    unittest.main()
