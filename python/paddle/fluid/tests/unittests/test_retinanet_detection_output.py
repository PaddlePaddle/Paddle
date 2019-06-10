#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License")
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import print_function
import unittest
import numpy as np
import math
import copy
from op_test import OpTest


def anchor_generator_in_python(input_feat, anchor_sizes, aspect_ratios,
                               variances, stride, offset):
    num_anchors = len(aspect_ratios) * len(anchor_sizes)
    layer_h = input_feat.shape[2]
    layer_w = input_feat.shape[3]
    out_dim = (layer_h, layer_w, num_anchors, 4)
    out_anchors = np.zeros(out_dim).astype('float32')

    for h_idx in range(layer_h):
        for w_idx in range(layer_w):
            x_ctr = (w_idx * stride[0]) + offset * (stride[0] - 1)
            y_ctr = (h_idx * stride[1]) + offset * (stride[1] - 1)
            idx = 0
            for r in range(len(aspect_ratios)):
                ar = aspect_ratios[r]
                for s in range(len(anchor_sizes)):
                    anchor_size = anchor_sizes[s]
                    area = stride[0] * stride[1]
                    area_ratios = area / ar
                    base_w = np.round(np.sqrt(area_ratios))
                    base_h = np.round(base_w * ar)
                    scale_w = anchor_size / stride[0]
                    scale_h = anchor_size / stride[1]
                    w = scale_w * base_w
                    h = scale_h * base_h
                    out_anchors[h_idx, w_idx, idx, :] = [
                        (x_ctr - 0.5 * (w - 1)), (y_ctr - 0.5 * (h - 1)),
                        (x_ctr + 0.5 * (w - 1)), (y_ctr + 0.5 * (h - 1))
                    ]
                    idx += 1

    # set the variance.
    out_var = np.tile(variances, (layer_h, layer_w, num_anchors, 1))
    out_anchors = out_anchors.astype('float32')
    out_var = out_var.astype('float32')
    return out_anchors, out_var


def iou(box_a, box_b, norm):
    """Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a + (norm == False)) * (xmax_a - xmin_a +
                                                    (norm == False))
    area_b = (ymax_b - ymin_b + (norm == False)) * (xmax_b - xmin_b +
                                                    (norm == False))
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa + (norm == False),
                     0.0) * max(yb - ya + (norm == False), 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)

    return iou_ratio


def nms(cls_dets, nms_threshold=0.05, eta=1.0):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        score_threshold: (float) The confidence thresh for filtering low
            confidence boxes.
        nms_threshold: (float) The overlap thresh for suppressing unnecessary
            boxes.
        top_k: (int) The maximum number of box preds to consider.
        eta: (float) The parameter for adaptive NMS.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    all_scores = np.zeros(len(cls_dets))
    for i in range(all_scores.shape[0]):
        all_scores[i] = cls_dets[i][4]
    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')

    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_indices.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(cls_dets[idx], cls_dets[kept_idx], False)
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold *= eta
    return selected_indices


def multiclass_nms(prediction, class_num, keep_top_k, nms_threshold):
    selected_indices = {}
    num_det = 0
    for c in range(class_num):
        cls_dets = prediction[c]
        indices = nms(cls_dets, nms_threshold)

        selected_indices[c] = indices
        num_det += len(indices)

    score_index = []
    for c, indices in selected_indices.items():
        for idx in indices:
            score_index.append((prediction[c][idx][4], c, idx))

    sorted_score_index = sorted(
        score_index, key=lambda tup: tup[0], reverse=True)
    if keep_top_k > -1 and num_det > keep_top_k:
        sorted_score_index = sorted_score_index[:keep_top_k]
        num_det = keep_top_k
    nmsed_outs = []
    for s, c, idx in sorted_score_index:
        xmin = prediction[c][idx][0]
        ymin = prediction[c][idx][1]
        xmax = prediction[c][idx][2]
        ymax = prediction[c][idx][3]
        nmsed_outs.append([c, s, xmin, ymin, xmax, ymax])

    return nmsed_outs, num_det


def retinanet_detection_out(boxes, scores, anchors, im_info, min_level,
                            max_level, score_threshold, nms_threshold,
                            nms_top_k, keep_top_k):
    class_num = scores.shape[1]
    total_cell_num = scores.shape[0]
    im_height, im_width, im_scale = im_info
    factors = 0
    for lvl in range(min_level, max_level + 1):
        factors += pow(2, max_level - lvl) * pow(2, max_level - lvl)
    coarsest_cell_num = total_cell_num / factors

    begin_idx = 0
    end_idx = 0
    all_scores = copy.deepcopy(scores)
    all_boxes = copy.deepcopy(boxes)
    all_anchors = copy.deepcopy(anchors)
    prediction = {}
    for lvl in range(min_level, max_level + 1):
        factor = pow(2, max_level - lvl)
        begin_idx = int(end_idx)
        end_idx = int(begin_idx + coarsest_cell_num * factor * factor)

        scores_per_level = all_scores[begin_idx:end_idx, :]
        scores_per_level = scores_per_level.flatten()
        bboxes_per_level = all_boxes[begin_idx:end_idx, :]
        bboxes_per_level = bboxes_per_level.flatten()
        anchors_per_level = all_anchors[begin_idx:end_idx, :]
        anchors_per_level = anchors_per_level.flatten()

        thresh = score_threshold if lvl < max_level else 0.0
        selected_indices = np.argwhere(scores_per_level > thresh)
        scores = scores_per_level[selected_indices]
        sorted_indices = np.argsort(-scores, axis=0, kind='mergesort')
        if nms_top_k > -1 and nms_top_k < sorted_indices.shape[0]:
            sorted_indices = sorted_indices[:nms_top_k]

        for i in range(sorted_indices.shape[0]):
            idx = selected_indices[sorted_indices[i]]
            idx = idx[0][0]
            a = idx / class_num
            c = idx % class_num
            box_offset = a * 4
            anchor_box_width = anchors_per_level[
                box_offset + 2] - anchors_per_level[box_offset] + 1
            anchor_box_height = anchors_per_level[
                box_offset + 3] - anchors_per_level[box_offset + 1] + 1
            anchor_box_center_x = anchors_per_level[
                box_offset] + anchor_box_width / 2
            anchor_box_center_y = anchors_per_level[box_offset +
                                                    1] + anchor_box_height / 2

            target_box_center_x = bboxes_per_level[
                box_offset] * anchor_box_width + anchor_box_center_x
            target_box_center_y = bboxes_per_level[
                box_offset + 1] * anchor_box_height + anchor_box_center_y
            target_box_width = math.exp(bboxes_per_level[box_offset +
                                                         2]) * anchor_box_width
            target_box_height = math.exp(bboxes_per_level[
                box_offset + 3]) * anchor_box_height

            pred_box_xmin = target_box_center_x - target_box_width / 2
            pred_box_ymin = target_box_center_y - target_box_height / 2
            pred_box_xmax = target_box_center_x + target_box_width / 2 - 1
            pred_box_ymax = target_box_center_y + target_box_height / 2 - 1

            pred_box_xmin = pred_box_xmin / im_scale
            pred_box_ymin = pred_box_ymin / im_scale
            pred_box_xmax = pred_box_xmax / im_scale
            pred_box_ymax = pred_box_ymax / im_scale

            pred_box_xmin = max(
                min(pred_box_xmin, np.round(im_width / im_scale) - 1), 0.)
            pred_box_ymin = max(
                min(pred_box_ymin, np.round(im_height / im_scale) - 1), 0.)
            pred_box_xmax = max(
                min(pred_box_xmax, np.round(im_width / im_scale) - 1), 0.)
            pred_box_ymax = max(
                min(pred_box_ymax, np.round(im_height / im_scale) - 1), 0.)

            if c not in prediction.keys():
                prediction[c] = []
            prediction[c].append([
                pred_box_xmin, pred_box_ymin, pred_box_xmax, pred_box_ymax,
                scores_per_level[idx]
            ])

    nmsed_outs, nmsed_num = multiclass_nms(prediction, class_num, keep_top_k,
                                           nms_threshold)
    return nmsed_outs, nmsed_num


def batched_retinanet_detection_out(boxes, scores, anchors, im_info, min_level,
                                    max_level, score_threshold, nms_threshold,
                                    nms_top_k, keep_top_k):
    batch_size = scores.shape[0]
    det_outs = []
    lod = []
    for n in range(batch_size):
        nmsed_outs, nmsed_num = retinanet_detection_out(
            boxes[n], scores[n], anchors, im_info[n], min_level, max_level,
            score_threshold, nms_threshold, nms_top_k, keep_top_k)
        if nmsed_num == 0:
            continue

        lod.append(nmsed_num)
        det_outs.extend(nmsed_outs)
    if len(lod) == 0:
        lod += [1]
    return det_outs, lod


class TestRetinanetDetectionOutOp1(OpTest):
    def set_argument(self):
        self.score_threshold = 0.05
        self.min_level = 3
        self.max_level = 7
        self.nms_threshold = 0.3
        self.nms_top_k = 1000
        self.keep_top_k = 200

        self.scales_per_octave = 3
        self.aspect_ratios = [1.0, 2.0, 0.5]
        self.anchor_scale = 4
        self.anchor_strides = [8, 16, 32, 64, 128]

        self.box_size = 4
        self.class_num = 80
        self.batch_size = 1
        self.input_channels = 20

    def init_test_input(self):
        anchor_num = len(self.aspect_ratios) * self.scales_per_octave
        num_levels = self.max_level - self.min_level + 1
        scores = []
        boxes = []
        anchors = []

        for i in range(num_levels):
            layer_h = 2**(num_levels - i)
            layer_w = 2**(num_levels - i)

            input_feat = np.random.random((self.batch_size, self.input_channels,
                                           layer_h, layer_w)).astype('float32')
            score = np.random.random(
                (self.batch_size, self.class_num * anchor_num, layer_h,
                 layer_w)).astype('float32')
            score = np.transpose(score, [0, 2, 3, 1])
            score = score.reshape((self.batch_size, -1, self.class_num))
            box = np.random.random((self.batch_size, self.box_size * anchor_num,
                                    layer_h, layer_w)).astype('float32')
            box = np.transpose(box, [0, 2, 3, 1])
            box = box.reshape((self.batch_size, -1, self.box_size))
            anchor_sizes = []
            for octave in range(self.scales_per_octave):
                anchor_sizes.append(
                    float(self.anchor_strides[i] * (2**octave)) /
                    float(self.scales_per_octave) * self.anchor_scale)
            anchor, var = anchor_generator_in_python(
                input_feat=input_feat,
                anchor_sizes=anchor_sizes,
                aspect_ratios=self.aspect_ratios,
                variances=[1.0, 1.0, 1.0, 1.0],
                stride=[self.anchor_strides[i], self.anchor_strides[i]],
                offset=0.5)
            anchor = np.reshape(anchor, [-1, 4])
            scores.append(score)
            boxes.append(box)
            anchors.append(anchor)

        self.anchor_list = np.concatenate(anchors, axis=0).astype('float32')
        self.score_list = np.concatenate(scores, axis=1).astype('float32')
        self.box_list = np.concatenate(boxes, axis=1).astype('float32')
        self.im_info = np.array([[256., 256., 1.5]]).astype(
            'float32')  #im_height, im_width, scale

    def setUp(self):
        self.set_argument()
        self.init_test_input()

        nmsed_outs, lod = batched_retinanet_detection_out(
            self.box_list, self.score_list, self.anchor_list, self.im_info,
            self.min_level, self.max_level, self.score_threshold,
            self.nms_threshold, self.nms_top_k, self.keep_top_k)
        nmsed_outs = [-1] if not nmsed_outs else nmsed_outs
        nmsed_outs = np.array(nmsed_outs).astype('float32')

        self.op_type = 'retinanet_detection_output'
        self.inputs = {
            'BBoxes': self.box_list,
            'Scores': self.score_list,
            'Anchors': self.anchor_list,
            'ImInfo': (self.im_info, [[1, ]])
        }
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'min_level': self.min_level,
            'max_level': self.max_level,
            'score_threshold': self.score_threshold,
            'nms_top_k': self.nms_top_k,
            'nms_threshold': self.nms_threshold,
            'keep_top_k': self.keep_top_k,
            'nms_eta': 1.,
        }

    def test_check_output(self):
        self.check_output()


class TestRetinanetDetectionOutOpNo2(TestRetinanetDetectionOutOp1):
    def set_argument(self):
        # Here set 2.0 to test the case there is no outputs.
        # In practical use, 0.0 < score_threshold < 1.0
        self.score_threshold = 2.0
        self.min_level = 3
        self.max_level = 7
        self.nms_threshold = 0.3
        self.nms_top_k = 1000
        self.keep_top_k = 200

        self.scales_per_octave = 3
        self.aspect_ratios = [1.0, 2.0, 0.5]
        self.anchor_scale = 4
        self.anchor_strides = [8, 16, 32, 64, 128]

        self.box_size = 4
        self.class_num = 80
        self.batch_size = 1
        self.input_channels = 20


class TestRetinanetDetectionOutOpNo3(TestRetinanetDetectionOutOp1):
    def set_argument(self):
        self.score_threshold = 0.05
        self.min_level = 2
        self.max_level = 5
        self.nms_threshold = 0.3
        self.nms_top_k = 1000
        self.keep_top_k = 200

        self.scales_per_octave = 3
        self.aspect_ratios = [1.0, 2.0, 0.5]
        self.anchor_scale = 4
        self.anchor_strides = [8, 16, 32, 64, 128]

        self.box_size = 4
        self.class_num = 80
        self.batch_size = 1
        self.input_channels = 20


class TestRetinanetDetectionOutOpNo4(TestRetinanetDetectionOutOp1):
    def set_argument(self):
        self.score_threshold = 0.05
        self.min_level = 3
        self.max_level = 7
        self.nms_threshold = 0.3
        self.nms_top_k = 100
        self.keep_top_k = 10

        self.scales_per_octave = 3
        self.aspect_ratios = [1.0, 2.0, 0.5]
        self.anchor_scale = 4
        self.anchor_strides = [8, 16, 32, 64, 128]

        self.box_size = 4
        self.class_num = 80
        self.batch_size = 1
        self.input_channels = 20


if __name__ == '__main__':
    unittest.main()
