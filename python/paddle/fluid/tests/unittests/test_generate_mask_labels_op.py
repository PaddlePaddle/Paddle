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

import unittest
import numpy as np
import sys
import math
import six
import paddle.fluid as fluid
from op_test import OpTest
'''
# Equivalent code
rles = mask_util.frPyObjects([segm], im_h, im_w)
mask = mask_util.decode(rles)
'''


def decode(cnts, m):
    v = 0
    mask = []
    for j in range(m):
        for k in range(cnts[j]):
            mask.append(v)
        v = 1 - v
    return mask


def poly2mask(xy, k, h, w):
    scale = 5.
    x = [int(scale * p + 0.5) for p in xy[::2]]
    x = x + [x[0]]
    y = [int(scale * p + 0.5) for p in xy[1::2]]
    y = y + [y[0]]
    m = sum([
        int(max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1]))) + int(1)
        for j in range(k)
    ])

    u, v = [], []
    for j in range(k):
        xs = x[j]
        xe = x[j + 1]
        ys = y[j]
        ye = y[j + 1]
        dx = abs(xe - xs)
        dy = abs(ys - ye)
        flip = (dx >= dy and xs > xe) or (dx < dy and ys > ye)
        if flip:
            xs, xe = xe, xs
            ys, ye = ye, ys

        if dx >= dy:
            if (dx == 0): assert ye - ys == 0
            s = 0 if dx == 0 else float(ye - ys) / dx
        else:
            if (dy == 0): assert xe - xs == 0
            s = 0 if dy == 0 else float(xe - xs) / dy

        if dx >= dy:
            ts = [dx - d if flip else d for d in range(dx + 1)]
            u.extend([xs + t for t in ts])
            v.extend([int(ys + s * t + .5) for t in ts])
        else:
            ts = [dy - d if flip else d for d in range(dy + 1)]
            v.extend([t + ys for t in ts])
            u.extend([int(xs + s * t + .5) for t in ts])

    k = len(u)
    x = np.zeros((k), np.int_)
    y = np.zeros((k), np.int_)
    m = 0
    for j in six.moves.xrange(1, k):
        if u[j] != u[j - 1]:
            xd = float(u[j] if (u[j] < u[j - 1]) else (u[j] - 1))
            xd = (xd + .5) / scale - .5
            if (math.floor(xd) != xd or xd < 0 or xd > (w - 1)):
                continue
            yd = float(v[j] if v[j] < v[j - 1] else v[j - 1])
            yd = (yd + .5) / scale - .5
            yd = math.ceil(0 if yd < 0 else (h if yd > h else yd))
            x[m] = int(xd)
            y[m] = int(yd)
            m += 1
    k = m
    a = [int(x[i] * h + y[i]) for i in range(k)]
    a.append(h * w)
    a.sort()
    b = [0] + a[:len(a) - 1]
    a = [c - d for (c, d) in zip(a, b)]

    k += 1
    b = [0 for i in range(k)]
    b[0] = a[0]
    m, j = 1, 1
    while (j < k):
        if a[j] > 0:
            b[m] = a[j]
            m += 1
            j += 1
        else:
            j += 1
            if (j < k):
                b[m - 1] += a[j]
                j += 1
    mask = decode(b, m)
    mask = np.array(mask, dtype=np.int_).reshape((w, h))
    mask = mask.transpose((1, 0))
    return mask


def polys_to_boxes(polys):
    """Convert a list of polygons into an array of tight bounding boxes."""
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]
    return boxes_from_polys


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


def polys_to_mask_wrt_box(polygons, box, M):
    """Convert from the COCO polygon segmentation format to a binary mask
    encoded as a 2D array of data type numpy.float32. The polygon segmentation
    is understood to be enclosed in the given box and rasterized to an M x M
    mask. The resulting mask is therefore of shape (M, M).
    """
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    mask = []
    for polygons in polygons_norm:
        assert polygons.shape[0] % 2 == 0
        k = polygons.shape[0] // 2
        mask.append(poly2mask(polygons, k, M, M))
    mask = np.array(mask)
    # Flatten in case polygons was a list
    mask = np.sum(mask, axis=0)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


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


def generate_mask_labels(num_classes, im_info, gt_classes, is_crowd,
                         label_int32, gt_polys, resolution, rois, roi_lod,
                         gt_lod):
    mask_rois = []
    roi_has_mask_int32 = []
    mask_int32 = []
    new_lod = []
    for i in range(len(im_info)):
        roi_s = roi_lod[i]
        roi_e = roi_lod[i + 1]
        gt_s = gt_lod[i]
        gt_e = gt_lod[i + 1]
        mask_blob = _sample_mask(num_classes, im_info[i], gt_classes[gt_s:gt_e],
                                 is_crowd[gt_s:gt_e], label_int32[roi_s:roi_e],
                                 gt_polys[i], resolution, rois[roi_s:roi_e])
        new_lod.append(mask_blob['mask_rois'].shape[0])
        mask_rois.append(mask_blob['mask_rois'])
        roi_has_mask_int32.append(mask_blob['roi_has_mask_int32'])
        mask_int32.append(mask_blob['mask_int32'])
    return mask_rois, roi_has_mask_int32, mask_int32, new_lod


def _sample_mask(
        num_classes,
        im_info,
        gt_classes,
        is_crowd,
        label_int32,
        gt_polys,  # [[[], []], []]
        resolution,
        rois):
    mask_blob = {}
    im_scale = im_info[2]
    sample_boxes = rois
    polys_gt_inds = np.where((gt_classes > 0) & (is_crowd == 0))[0]
    polys_gt = [gt_polys[i] for i in polys_gt_inds]
    boxes_from_polys = polys_to_boxes(polys_gt)

    fg_inds = np.where(label_int32 > 0)[0]
    roi_has_mask = fg_inds.copy()
    if fg_inds.shape[0] > 0:
        mask_class_labels = label_int32[fg_inds]
        masks = np.zeros((fg_inds.shape[0], resolution**2), dtype=np.int32)
        rois_fg = sample_boxes[fg_inds]
        overlaps_bbfg_bbpolys = bbox_overlaps(
            rois_fg.astype(np.float32), boxes_from_polys.astype(np.float32))
        fg_polys_inds = np.argmax(overlaps_bbfg_bbpolys, axis=1)
        for i in range(rois_fg.shape[0]):
            fg_polys_ind = fg_polys_inds[i]
            poly_gt = polys_gt[fg_polys_ind]
            roi_fg = rois_fg[i]
            mask = polys_to_mask_wrt_box(poly_gt, roi_fg, resolution)
            mask = np.array(mask > 0, dtype=np.int32)
            masks[i, :] = np.reshape(mask, resolution**2)
    else:
        bg_inds = np.where(label_int32 == 0)[0]
        rois_fg = sample_boxes[bg_inds[0]].reshape((1, -1))
        masks = -np.ones((1, resolution**2), dtype=np.int32)
        mask_class_labels = np.zeros((1, ))
        roi_has_mask = np.append(roi_has_mask, 0)
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


class TestGenerateMaskLabels(OpTest):

    def set_data(self):
        self.init_test_case()
        self.make_generate_proposal_labels_out()
        self.generate_gt_polys()
        self.generate_groundtruth()
        self.init_test_output()
        self.inputs = {
            'ImInfo': self.im_info,
            'GtClasses': (self.gt_classes.astype(np.int32), self.gt_lod),
            'IsCrowd': (self.is_crowd.astype(np.int32), self.gt_lod),
            'LabelsInt32': (self.label_int32.astype(np.int32), self.rois_lod),
            'GtSegms': (self.gt_polys.astype(np.float32), self.masks_lod),
            'Rois': (self.rois.astype(np.float32), self.rois_lod)
        }
        self.attrs = {
            'num_classes': self.num_classes,
            'resolution': self.resolution
        }
        self.outputs = {
            'MaskRois': (self.mask_rois, [self.new_lod]),
            'RoiHasMaskInt32': (self.roi_has_mask_int32, [self.new_lod]),
            'MaskInt32': (self.mask_int32, [self.new_lod])
        }

    def init_test_case(self):
        self.num_classes = 81
        self.resolution = 14
        self.batch_size = 2
        self.batch_size_per_im = 64
        self.images_shape = [100, 200]
        np.random.seed(0)

    def make_generate_proposal_labels_out(self):
        rois = []
        self.rois_lod = [[]]
        self.label_int32 = []
        for bno in range(self.batch_size):
            self.rois_lod[0].append(self.batch_size_per_im)
            for i in range(self.batch_size_per_im):
                xywh = np.random.rand(4)
                xy1 = xywh[0:2] * 2
                wh = xywh[2:4] * (self.images_shape[0] - xy1)
                xy2 = xy1 + wh
                roi = [xy1[0], xy1[1], xy2[0], xy2[1]]
                rois.append(roi)
        self.rois = np.array(rois).astype("float32")
        for idx, roi_num in enumerate(self.rois_lod[0]):
            for roi_id in range(roi_num):
                class_id = np.random.random_integers(self.num_classes - 1)
                if idx == 0:
                    # set an image with no foreground, to test the empty case
                    self.label_int32.append(0)
                else:
                    self.label_int32.append(class_id)
        label_np = np.array(self.label_int32)
        self.label_int32 = label_np[:, np.newaxis]

    def generate_gt_polys(self):
        h, w = self.images_shape[0:2]
        self.gt_polys = []
        self.gt_polys_list = []
        max_gt = 4
        max_poly_num = 5
        min_poly_size = 4
        max_poly_size = 16
        lod0 = []
        lod1 = []
        lod2 = []
        for i in range(self.batch_size):
            gt_num = np.random.randint(1, high=max_gt, size=1)[0]
            lod0.append(gt_num)
            ptss = []
            for i in range(gt_num):
                poly_num = np.random.randint(1, max_poly_num, size=1)[0]
                lod1.append(poly_num)
                pts = []
                for j in range(poly_num):
                    poly_size = np.random.randint(min_poly_size,
                                                  max_poly_size,
                                                  size=1)[0]
                    x = np.random.rand(poly_size, 1) * w
                    y = np.random.rand(poly_size, 1) * h
                    xy = np.concatenate((x, y), axis=1)
                    pts.append(xy.flatten().tolist())
                    self.gt_polys.extend(xy.flatten().tolist())
                    lod2.append(poly_size)
                ptss.append(pts)
            self.gt_polys_list.append(ptss)
        self.masks_lod = [lod0, lod1, lod2]
        self.gt_lod = [lod0]
        self.gt_polys = np.array(self.gt_polys).astype('float32').reshape(-1, 2)

    def generate_groundtruth(self):
        self.im_info = []
        self.gt_classes = []
        self.is_crowd = []
        for roi_num in self.gt_lod[0]:
            self.im_info.append(self.images_shape + [1.0])
            for roi_id in range(roi_num):
                class_id = np.random.random_integers(self.num_classes - 1)
                self.gt_classes.append(class_id)
                self.is_crowd.append(0)
        self.im_info = np.array(self.im_info).astype(np.float32)
        gt_classes_np = np.array(self.gt_classes)
        self.gt_classes = gt_classes_np[:, np.newaxis]
        is_crowd_np = np.array(self.is_crowd)
        self.is_crowd = is_crowd_np[:, np.newaxis]

    def init_test_output(self):
        roi_lod = trans_lod(self.rois_lod[0])
        gt_lod = trans_lod(self.gt_lod[0])
        outs = generate_mask_labels(self.num_classes, self.im_info,
                                    self.gt_classes, self.is_crowd,
                                    self.label_int32, self.gt_polys_list,
                                    self.resolution, self.rois, roi_lod, gt_lod)
        self.mask_rois = outs[0]
        self.roi_has_mask_int32 = outs[1]
        self.mask_int32 = outs[2]
        self.new_lod = outs[3]

        self.mask_rois = np.vstack(self.mask_rois)
        self.roi_has_mask_int32 = np.hstack(self.roi_has_mask_int32)[:,
                                                                     np.newaxis]
        self.mask_int32 = np.vstack(self.mask_int32)

    def setUp(self):
        self.op_type = "generate_mask_labels"
        self.set_data()

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
