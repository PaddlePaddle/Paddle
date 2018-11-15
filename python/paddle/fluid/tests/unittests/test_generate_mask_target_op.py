# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import math
import sys
import math
import paddle.fluid as fluid
from op_test import OpTest


def masks_to_boxes(masks):
    """Convert a list of masks into an array of tight bounding boxes."""
    boxes_from_masks = np.zeros((len(masks), 4), dtype=np.float32)
    for i in range(len(masks)):
        mask = masks[i]
        mask_loc = np.where(mask > 0)
        xmin = np.min(mask_loc[1])
        xmax = np.max(mask_loc[1])
        ymin = np.min(mask_loc[0])
        ymax = np.max(mask_loc[0])
        boxes_from_masks[i, :] = np.array([xmin, ymin, xmax, ymax],\
                                                      dtype=np.int32)
    return boxes_from_masks


def bbox_overlaps(boxes, query_boxes):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + 1) *\
                   (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        for n in range(N):
            iw = min(boxes[n, 2], query_boxes[k, 2]) -\
                 max(boxes[n, 0], query_boxes[k, 0]) + 1
            if iw > 0:
                ih = min(boxes[n, 3], query_boxes[k, 3]) -\
                     max(boxes[n, 1], query_boxes[k, 1]) + 1
                if ih > 0:
                    ua = float(
                         (boxes[n, 2] - boxes[n, 0] + 1) *\
                         (boxes[n, 3] - boxes[n, 1] + 1) +\
                         box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def crop_and_resize(mask_gt, roi, resolution):
    result = np.zeros((resolution, resolution))
    w = roi[2] - roi[0]
    h = roi[3] - roi[1]
    w = np.maximum(w, 1)
    h = np.maximum(h, 1)
    for i in range(resolution):
        for j in range(resolution):
            x = int(i / float(resolution) * w + roi[0])
            y = int(j / float(resolution) * h + roi[1])
            result[j, i] = mask_gt[y, x] > 0
    return result


def expand_mask_targets(masks, mask_class_labels, resolution, num_classes):
    """Expand masks from shape (#masks, resolution ** 2)
    to (#masks, #classes * resolution ** 2) to encode class
    specific mask targets.
    """
    assert masks.shape[0] == mask_class_labels.shape[0]

    # Target values of -1 are "don't care" / ignore labels
    mask_targets = -np.ones(
        (masks.shape[0], num_classes * resolution**2), dtype=np.int32)
    for i in range(masks.shape[0]):
        cls = int(mask_class_labels[i])
        start = resolution**2 * cls
        end = start + resolution**2
        # Ignore background instance
        # (only happens when there is no fg samples in an image)
        if cls > 0:
            mask_targets[i, start:end] = masks[i, :]
    return mask_targets


def generate_mask_target(num_classes, im_info, gt_classes, is_crowd,
                         label_int32, gt_segms, resolution, rois, roi_lod,
                         mask_lod):
    mask_rois = []
    roi_has_mask_int32 = []
    mask_int32 = []
    new_lod = []
    for im_i in range(len(im_info)):
        roi_start = roi_lod[im_i]
        roi_end = roi_lod[im_i + 1]
        mask_start = mask_lod[im_i]
        mask_end = mask_lod[im_i + 1]
        mask_blob = _sample_mask(
            num_classes, im_info[im_i], gt_classes[mask_start:mask_end],
            is_crowd[mask_start:mask_end], label_int32[roi_start:roi_end],
            gt_segms[mask_start:mask_end], resolution, rois[roi_start:roi_end])
        new_lod.append(mask_blob['mask_rois'].shape[0])
        mask_rois.append(mask_blob['mask_rois'])
        roi_has_mask_int32.append(mask_blob['roi_has_mask_int32'])
        mask_int32.append(mask_blob['mask_int32'])
    return mask_rois, roi_has_mask_int32, mask_int32, new_lod


def _sample_mask(num_classes, im_info, gt_classes, is_crowd, label_int32,
                 gt_segms, resolution, rois):
    mask_blob = {}
    im_scale = im_info[2]
    sample_boxes = rois / im_scale
    mask_gt_inds = np.where((gt_classes > 0) & (is_crowd == 0))[0]
    masks_gt = []
    masks_gt = [gt_segms[i] for i in mask_gt_inds]
    boxes_from_masks = masks_to_boxes(masks_gt)
    fg_inds = np.where(label_int32 > 0)[0]
    roi_has_mask = fg_inds.copy()
    if fg_inds.shape[0] > 0:
        mask_class_labels = label_int32[fg_inds]
        masks = np.zeros((fg_inds.shape[0], resolution**2), dtype=np.int32)
        rois_fg = sample_boxes[fg_inds]
        overlaps_bbfg_bbmasks = bbox_overlaps(
            rois_fg.astype(np.float32), boxes_from_masks.astype(np.float32))
        fg_masks_inds = np.argmax(overlaps_bbfg_bbmasks, axis=1)
        for i in range(rois_fg.shape[0]):
            fg_masks_ind = fg_masks_inds[i]
            mask_gt = masks_gt[fg_masks_ind]
            roi_fg = rois_fg[i]
            mask = crop_and_resize(mask_gt, roi_fg, resolution)
            mask = np.array(mask > 0, dtype=np.int32)
            masks[i, :] = np.reshape(mask, resolution**2)
    else:
        bg_inds = np.where(label_int32 == 0)[0]
        rois_fg = sampled_boxes[bg_inds[0]].reshape((1, -1))
        masks = -np.ones((1, resolution**2), dtype=np.int32)
        mask_class_labels = np.zeros((1, ))
        np.append(roi_has_mask, 0)
    masks = expand_mask_targets(masks, mask_class_labels, resolution,
                                num_classes)
    rois_fg *= im_scale
    mask_blob['mask_rois'] = rois_fg
    mask_blob['roi_has_mask_int32'] = roi_has_mask
    mask_blob['mask_int32'] = masks
    return mask_blob


def trans_lod(lod):
    new_lod = [0]
    for i in range(len(lod)):
        new_lod.append(lod[i] + new_lod[i])
    return new_lod


class TestGenerateMaskTarget(OpTest):
    def set_data(self):
        self.init_test_case()
        self.make_generate_proposal_labels_out()
        self.generate_gt_segms()
        self.generate_groundtruth()
        self.init_test_output()
        self.inputs = {
            'ImInfo': self.im_info,
            'GtClasses': (self.gt_classes, self.masks_lod),
            'IsCrowd': (self.is_crowd, self.masks_lod),
            'LabelsInt32': (self.label_int32, self.rois_lod),
            'GtSegms': (self.gt_segms, self.masks_lod),
            'Rois': (self.rois, self.rois_lod)
        }
        self.attrs = {
            'num_classes': self.num_classes,
            'resolution': self.resolution
        }
        self.outputs = {
            'MaskRois': (self.mask_rois, self.new_lod),
            'RoiHasMaskInt32': (self.roi_has_mask_int32, self.new_lod),
            'MaskInt32': (self.mask_int32, self.new_lod)
        }

    def init_test_case(self):
        self.num_classes = 5
        self.resolution = 14
        self.batch_size = 4
        self.images_shape = [16, 16]
        np.random.seed(0)

    def make_generate_proposal_labels_out(self):
        rois = []
        self.rois_lod = [[]]
        self.label_int32 = []
        for bno in range(self.batch_size):
            self.rois_lod[0].append(2 * (bno + 1))
            for i in range(2 * (bno + 1)):
                xywh = np.random.rand(4)
                xy1 = xywh[0:2] * 2
                wh = xywh[2:4] * (self.images_shape[0] - xy1)
                xy2 = xy1 + wh
                roi = [xy1[0], xy1[1], xy2[0], xy2[1]]
                rois.append(roi)
        self.rois_num = len(rois)
        self.rois = np.array(rois).astype("float32")
        for roi_num in self.rois_lod[0]:
            for roi_id in range(roi_num):
                class_id = np.random.random_integers(self.num_classes - 1)
                self.label_int32.append(class_id)
        label_np = np.array(self.label_int32)
        self.label_int32 = label_np[:, np.newaxis]

    def generate_gt_segms(self):
        self.gt_segms = np.zeros((10, self.images_shape[0], \
                                  self.images_shape[1]), dtype=np.int8)
        self.masks_lod = [[]]
        mask_id = 0
        for bno in range(self.batch_size):
            self.masks_lod[0].append(bno + 1)
            for i in range(bno + 1):
                xywh = np.random.rand(4)
                xy1 = xywh[0:2] * 2
                wh = xywh[2:4] * (self.images_shape[0] - xy1)
                xy2 = xy1 + wh
                self.gt_segms[mask_id, int(xy1[0]):int(math.ceil(xy2[0])),\
                              int(xy1[1]):int(math.ceil(xy2[1]))] = 1
                mask_id += 1

    def generate_groundtruth(self):
        self.im_info = []
        self.gt_classes = []
        self.is_crowd = []
        for roi_num in self.masks_lod[0]:
            self.im_info.append(self.images_shape + [1.0])
            for roi_id in range(roi_num):
                class_id = np.random.random_integers(self.num_classes - 1)
                self.gt_classes.append(class_id)
                self.is_crowd.append(0)
        self.im_info = np.array(self.im_info)
        gt_classes_np = np.array(self.gt_classes)
        self.gt_classes = gt_classes_np[:, np.newaxis]
        is_crowd_np = np.array(self.is_crowd)
        self.is_crowd = is_crowd_np[:, np.newaxis]

    def init_test_output(self):
        roi_lod = trans_lod(self.rois_lod[0])
        mask_lod = trans_lod(self.masks_lod[0])
        self.mask_rois, self.roi_has_mask_int32, self.mask_int32, \
        self.new_lod = generate_mask_target(self.num_classes, self.im_info,
                   self.gt_classes, self.is_crowd,
                   self.label_int32, self.gt_segms,
                   self.resolution, self.rois,
                   roi_lod, mask_lod)
        self.mask_rois = np.vstack(self.mask_rois)
        self.roi_has_mask_int32 = np.hstack(self.roi_has_mask_int32)
        self.mask_int32 = np.vstack(self.mask_int32)

    def setUp(self):
        self.op_type = "generate_mask_target"
        self.set_data()

    def test_check_output(self):
        self.check_output()
