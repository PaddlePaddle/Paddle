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
from op_test import OpTest
from test_anchor_generator_op import anchor_generator_in_python


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


def _generate_groundtruth(images_shape, class_nums, gt_nums):
    ground_truth = []
    gts_lod = []
    num_gts = 0
    for i, image_shape in enumerate(images_shape):
        # Avoid background
        gt_classes = np.random.randint(
            low=1, high=class_nums, size=gt_nums
        ).astype(np.int32)
        gt_boxes = _generate_boxes(image_shape, gt_nums)
        is_crowd = np.zeros((gt_nums), dtype=np.int32)
        is_crowd[0] = 1
        ground_truth.append(
            {'gt_classes': gt_classes, 'boxes': gt_boxes, 'is_crowd': is_crowd}
        )
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
    boxes[:, [0, 2]] = np.minimum(
        width - 1.0, np.maximum(0.0, boxes[:, [0, 2]])
    )
    boxes[:, [1, 3]] = np.minimum(
        height - 1.0, np.maximum(0.0, boxes[:, [1, 3]])
    )
    return boxes.astype(np.float32)


def rpn_target_assign(
    anchor_by_gt_overlap,
    rpn_batch_size_per_im,
    rpn_positive_overlap,
    rpn_negative_overlap,
    rpn_fg_fraction,
    use_random=True,
):
    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    anchor_to_gt_max = anchor_by_gt_overlap[
        np.arange(anchor_by_gt_overlap.shape[0]), anchor_to_gt_argmax
    ]

    gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    gt_to_anchor_max = anchor_by_gt_overlap[
        gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])
    ]
    anchors_with_max_overlap = np.where(
        anchor_by_gt_overlap == gt_to_anchor_max
    )[0]

    labels = np.ones((anchor_by_gt_overlap.shape[0],), dtype=np.int32) * -1
    labels[anchors_with_max_overlap] = 1
    labels[anchor_to_gt_max >= rpn_positive_overlap] = 1

    num_fg = int(rpn_fg_fraction * rpn_batch_size_per_im)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg and use_random:
        disable_inds = np.random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False
        )
    else:
        disable_inds = fg_inds[num_fg:]

    labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]
    bbox_inside_weight = np.zeros((len(fg_inds), 4), dtype=np.float32)

    num_bg = rpn_batch_size_per_im - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < rpn_negative_overlap)[0]
    if len(bg_inds) > num_bg and use_random:
        enable_inds = bg_inds[np.random.randint(len(bg_inds), size=num_bg)]
    else:
        enable_inds = bg_inds[:num_bg]

    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0

    bbox_inside_weight[fake_num:, :] = 1
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    loc_index = np.hstack([fg_fake_inds, fg_inds])
    score_index = np.hstack([fg_inds, bg_inds])
    labels = labels[score_index]
    assert not np.any(labels == -1), "Wrong labels with -1"

    gt_inds = anchor_to_gt_argmax[loc_index]

    return loc_index, score_index, labels, gt_inds, bbox_inside_weight


def get_anchor(n, c, h, w):
    input_feat = np.random.random((n, c, h, w)).astype('float32')
    anchors, _ = anchor_generator_in_python(
        input_feat=input_feat,
        anchor_sizes=[32.0, 64.0],
        aspect_ratios=[0.5, 1.0],
        variances=[1.0, 1.0, 1.0, 1.0],
        stride=[16.0, 16.0],
        offset=0.5,
    )
    return anchors


def rpn_target_assign_in_python(
    all_anchors,
    gt_boxes,
    is_crowd,
    im_info,
    lod,
    rpn_straddle_thresh,
    rpn_batch_size_per_im,
    rpn_positive_overlap,
    rpn_negative_overlap,
    rpn_fg_fraction,
    use_random=True,
):
    anchor_num = all_anchors.shape[0]
    batch_size = len(lod) - 1
    for i in range(batch_size):
        im_height = im_info[i][0]
        im_width = im_info[i][1]
        im_scale = im_info[i][2]
        if rpn_straddle_thresh >= 0:
            # Only keep anchors inside the image by a margin of straddle_thresh
            inds_inside = np.where(
                (all_anchors[:, 0] >= -rpn_straddle_thresh)
                & (all_anchors[:, 1] >= -rpn_straddle_thresh)
                & (all_anchors[:, 2] < im_width + rpn_straddle_thresh)
                & (all_anchors[:, 3] < im_height + rpn_straddle_thresh)
            )[0]
            # keep only inside anchors
            inside_anchors = all_anchors[inds_inside, :]
        else:
            inds_inside = np.arange(all_anchors.shape[0])
            inside_anchors = all_anchors

        b, e = lod[i], lod[i + 1]
        gt_boxes_slice = gt_boxes[b:e, :] * im_scale
        is_crowd_slice = is_crowd[b:e]

        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_boxes_slice = gt_boxes_slice[not_crowd_inds]
        iou = _bbox_overlaps(inside_anchors, gt_boxes_slice)

        (
            loc_inds,
            score_inds,
            labels,
            gt_inds,
            bbox_inside_weight,
        ) = rpn_target_assign(
            iou,
            rpn_batch_size_per_im,
            rpn_positive_overlap,
            rpn_negative_overlap,
            rpn_fg_fraction,
            use_random,
        )
        # unmap to all anchor
        loc_inds = inds_inside[loc_inds]
        score_inds = inds_inside[score_inds]

        sampled_gt = gt_boxes_slice[gt_inds]
        sampled_anchor = all_anchors[loc_inds]
        box_deltas = _box_to_delta(
            sampled_anchor, sampled_gt, [1.0, 1.0, 1.0, 1.0]
        )

        if i == 0:
            loc_indexes = loc_inds
            score_indexes = score_inds
            tgt_labels = labels
            tgt_bboxes = box_deltas
            bbox_inside_weights = bbox_inside_weight
        else:
            loc_indexes = np.concatenate(
                [loc_indexes, loc_inds + i * anchor_num]
            )
            score_indexes = np.concatenate(
                [score_indexes, score_inds + i * anchor_num]
            )
            tgt_labels = np.concatenate([tgt_labels, labels])
            tgt_bboxes = np.vstack([tgt_bboxes, box_deltas])
            bbox_inside_weights = np.vstack(
                [bbox_inside_weights, bbox_inside_weight]
            )

    return (
        loc_indexes,
        score_indexes,
        tgt_bboxes,
        tgt_labels,
        bbox_inside_weights,
    )


def retinanet_target_assign(
    anchor_by_gt_overlap, gt_labels, positive_overlap, negative_overlap
):
    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    anchor_to_gt_max = anchor_by_gt_overlap[
        np.arange(anchor_by_gt_overlap.shape[0]), anchor_to_gt_argmax
    ]

    gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    gt_to_anchor_max = anchor_by_gt_overlap[
        gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])
    ]
    anchors_with_max_overlap = np.where(
        anchor_by_gt_overlap == gt_to_anchor_max
    )[0]

    labels = np.ones((anchor_by_gt_overlap.shape[0],), dtype=np.int32) * -1
    labels[anchors_with_max_overlap] = 1
    labels[anchor_to_gt_max >= positive_overlap] = 1

    fg_inds = np.where(labels == 1)[0]
    bbox_inside_weight = np.zeros((len(fg_inds), 4), dtype=np.float32)

    bg_inds = np.where(anchor_to_gt_max < negative_overlap)[0]
    enable_inds = bg_inds

    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    labels[enable_inds] = 0

    bbox_inside_weight[fake_num:, :] = 1
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    loc_index = np.hstack([fg_fake_inds, fg_inds])
    score_index = np.hstack([fg_inds, bg_inds])
    score_index_tmp = np.hstack([fg_inds])
    labels = labels[score_index]

    gt_inds = anchor_to_gt_argmax[loc_index]
    label_inds = anchor_to_gt_argmax[score_index_tmp]
    labels[0 : len(fg_inds)] = np.squeeze(gt_labels[label_inds])
    fg_num = len(fg_fake_inds) + len(fg_inds) + 1
    assert not np.any(labels == -1), "Wrong labels with -1"
    return loc_index, score_index, labels, gt_inds, bbox_inside_weight, fg_num


def retinanet_target_assign_in_python(
    all_anchors,
    gt_boxes,
    gt_labels,
    is_crowd,
    im_info,
    lod,
    positive_overlap,
    negative_overlap,
):
    anchor_num = all_anchors.shape[0]
    batch_size = len(lod) - 1
    for i in range(batch_size):
        im_scale = im_info[i][2]

        inds_inside = np.arange(all_anchors.shape[0])
        inside_anchors = all_anchors
        b, e = lod[i], lod[i + 1]
        gt_boxes_slice = gt_boxes[b:e, :] * im_scale
        gt_labels_slice = gt_labels[b:e, :]
        is_crowd_slice = is_crowd[b:e]

        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_boxes_slice = gt_boxes_slice[not_crowd_inds]
        gt_labels_slice = gt_labels_slice[not_crowd_inds]
        iou = _bbox_overlaps(inside_anchors, gt_boxes_slice)

        (
            loc_inds,
            score_inds,
            labels,
            gt_inds,
            bbox_inside_weight,
            fg_num,
        ) = retinanet_target_assign(
            iou, gt_labels_slice, positive_overlap, negative_overlap
        )
        # unmap to all anchor
        loc_inds = inds_inside[loc_inds]
        score_inds = inds_inside[score_inds]

        sampled_gt = gt_boxes_slice[gt_inds]
        sampled_anchor = all_anchors[loc_inds]
        box_deltas = _box_to_delta(
            sampled_anchor, sampled_gt, [1.0, 1.0, 1.0, 1.0]
        )

        if i == 0:
            loc_indexes = loc_inds
            score_indexes = score_inds
            tgt_labels = labels
            tgt_bboxes = box_deltas
            bbox_inside_weights = bbox_inside_weight
            fg_nums = [[fg_num]]
        else:
            loc_indexes = np.concatenate(
                [loc_indexes, loc_inds + i * anchor_num]
            )
            score_indexes = np.concatenate(
                [score_indexes, score_inds + i * anchor_num]
            )
            tgt_labels = np.concatenate([tgt_labels, labels])
            tgt_bboxes = np.vstack([tgt_bboxes, box_deltas])
            bbox_inside_weights = np.vstack(
                [bbox_inside_weights, bbox_inside_weight]
            )
            fg_nums = np.concatenate([fg_nums, [[fg_num]]])

    return (
        loc_indexes,
        score_indexes,
        tgt_bboxes,
        tgt_labels,
        bbox_inside_weights,
        fg_nums,
    )


class TestRpnTargetAssignOp(OpTest):
    def setUp(self):
        n, c, h, w = 2, 4, 14, 14
        all_anchors = get_anchor(n, c, h, w)
        gt_num = 10
        all_anchors = all_anchors.reshape(-1, 4)
        anchor_num = all_anchors.shape[0]

        images_shape = [[64, 64], [64, 64]]
        # images_shape = [[64, 64]]
        groundtruth, lod = _generate_groundtruth(images_shape, 3, 4)
        lod = [0, 4, 8]
        # lod = [0, 4]

        im_info = np.ones((len(images_shape), 3)).astype(np.float32)
        for i in range(len(images_shape)):
            im_info[i, 0] = images_shape[i][0]
            im_info[i, 1] = images_shape[i][1]
            im_info[i, 2] = 0.8  # scale
        gt_boxes = np.vstack([v['boxes'] for v in groundtruth])
        is_crowd = np.hstack([v['is_crowd'] for v in groundtruth])

        all_anchors = all_anchors.astype('float32')
        gt_boxes = gt_boxes.astype('float32')

        rpn_straddle_thresh = 0.0
        rpn_batch_size_per_im = 256
        rpn_positive_overlap = 0.7
        rpn_negative_overlap = 0.3
        rpn_fg_fraction = 0.5
        use_random = False

        (
            loc_index,
            score_index,
            tgt_bbox,
            labels,
            bbox_inside_weights,
        ) = rpn_target_assign_in_python(
            all_anchors,
            gt_boxes,
            is_crowd,
            im_info,
            lod,
            rpn_straddle_thresh,
            rpn_batch_size_per_im,
            rpn_positive_overlap,
            rpn_negative_overlap,
            rpn_fg_fraction,
            use_random,
        )
        labels = labels[:, np.newaxis]

        self.op_type = "rpn_target_assign"
        self.inputs = {
            'Anchor': all_anchors,
            'GtBoxes': (gt_boxes, [[4, 4]]),
            'IsCrowd': (is_crowd, [[4, 4]]),
            'ImInfo': (im_info, [[1, 1]]),
        }
        self.attrs = {
            'rpn_batch_size_per_im': rpn_batch_size_per_im,
            'rpn_straddle_thresh': rpn_straddle_thresh,
            'rpn_positive_overlap': rpn_positive_overlap,
            'rpn_negative_overlap': rpn_negative_overlap,
            'rpn_fg_fraction': rpn_fg_fraction,
            'use_random': use_random,
        }
        self.outputs = {
            'LocationIndex': loc_index.astype('int32'),
            'ScoreIndex': score_index.astype('int32'),
            'TargetBBox': tgt_bbox.astype('float32'),
            'TargetLabel': labels.astype('int32'),
            'BBoxInsideWeight': bbox_inside_weights.astype('float32'),
        }

    def test_check_output(self):
        self.check_output()


class TestRetinanetTargetAssignOp(OpTest):
    def setUp(self):
        n, c, h, w = 2, 4, 14, 14
        all_anchors = get_anchor(n, c, h, w)
        gt_num = 10
        all_anchors = all_anchors.reshape(-1, 4)
        anchor_num = all_anchors.shape[0]

        images_shape = [[64, 64], [64, 64]]
        groundtruth, lod = _generate_groundtruth(images_shape, 3, 4)
        lod = [0, 4, 8]

        im_info = np.ones((len(images_shape), 3)).astype(np.float32)
        for i in range(len(images_shape)):
            im_info[i, 0] = images_shape[i][0]
            im_info[i, 1] = images_shape[i][1]
            im_info[i, 2] = 0.8  # scale
        gt_boxes = np.vstack([v['boxes'] for v in groundtruth])
        is_crowd = np.hstack([v['is_crowd'] for v in groundtruth])
        gt_labels = np.vstack(
            [
                v['gt_classes'].reshape(len(v['gt_classes']), 1)
                for v in groundtruth
            ]
        )
        gt_labels = gt_labels.reshape(len(gt_labels), 1)
        all_anchors = all_anchors.astype('float32')
        gt_boxes = gt_boxes.astype('float32')
        gt_labels = gt_labels.astype('int32')

        positive_overlap = 0.5
        negative_overlap = 0.4

        (
            loc_index,
            score_index,
            tgt_bbox,
            labels,
            bbox_inside_weights,
            fg_num,
        ) = retinanet_target_assign_in_python(
            all_anchors,
            gt_boxes,
            gt_labels,
            is_crowd,
            im_info,
            lod,
            positive_overlap,
            negative_overlap,
        )
        labels = labels[:, np.newaxis]
        self.op_type = "retinanet_target_assign"
        self.inputs = {
            'Anchor': all_anchors,
            'GtBoxes': (gt_boxes, [[4, 4]]),
            'GtLabels': (gt_labels, [[4, 4]]),
            'IsCrowd': (is_crowd, [[4, 4]]),
            'ImInfo': (im_info, [[1, 1]]),
        }
        self.attrs = {
            'positive_overlap': positive_overlap,
            'negative_overlap': negative_overlap,
        }
        self.outputs = {
            'LocationIndex': loc_index.astype('int32'),
            'ScoreIndex': score_index.astype('int32'),
            'TargetBBox': tgt_bbox.astype('float32'),
            'TargetLabel': labels.astype('int32'),
            'BBoxInsideWeight': bbox_inside_weights.astype('float32'),
            'ForegroundNumber': fg_num.astype('int32'),
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
