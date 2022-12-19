#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import unittest

import numpy as np
from op_test import OpTest
from test_multiclass_nms_op import iou


def weight_merge(box1, box2, score1, score2):
    for i in range(len(box1)):
        box2[i] = (box1[i] * score1 + box2[i] * score2) / (score1 + score2)


def nms(
    boxes,
    scores,
    score_threshold,
    nms_threshold,
    top_k=200,
    normalized=True,
    eta=1.0,
):
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
    index = -1
    for i in range(boxes.shape[0]):
        if (
            index > -1
            and iou(boxes[i], boxes[index], normalized) > nms_threshold
        ):
            weight_merge(boxes[i], boxes[index], scores[i], scores[index])
            scores[index] += scores[i]
            scores[i] = score_threshold - 1.0
        else:
            index = i

    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()

    selected_indices = np.argwhere(all_scores > score_threshold)
    selected_indices = selected_indices.flatten()
    all_scores = all_scores[selected_indices]

    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')
    sorted_scores = all_scores[sorted_indices]
    sorted_indices = selected_indices[sorted_indices]

    if top_k > -1 and top_k < sorted_indices.shape[0]:
        sorted_indices = sorted_indices[:top_k]
        sorted_scores = sorted_scores[:top_k]

    selected_indices = []
    adaptive_threshold = nms_threshold
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx], normalized)
                keep = True if overlap <= adaptive_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
        if keep and eta < 1 and adaptive_threshold > 0.5:
            adaptive_threshold *= eta
    return selected_indices


def multiclass_nms(
    boxes,
    scores,
    background,
    score_threshold,
    nms_threshold,
    nms_top_k,
    keep_top_k,
    normalized,
    shared,
):
    if shared:
        class_num = scores.shape[0]
        priorbox_num = scores.shape[1]
    else:
        box_num = scores.shape[0]
        class_num = scores.shape[1]

    selected_indices = {}
    num_det = 0
    for c in range(class_num):
        if c == background:
            continue
        if shared:
            indices = nms(
                boxes,
                scores[c],
                score_threshold,
                nms_threshold,
                nms_top_k,
                normalized,
            )
        else:
            indices = nms(
                boxes[:, c, :],
                scores[:, c],
                score_threshold,
                nms_threshold,
                nms_top_k,
                normalized,
            )
        selected_indices[c] = indices
        num_det += len(indices)

    if keep_top_k > -1 and num_det > keep_top_k:
        score_index = []
        for c, indices in selected_indices.items():
            for idx in indices:
                if shared:
                    score_index.append((scores[c][idx], c, idx))
                else:
                    score_index.append((scores[idx][c], c, idx))

        sorted_score_index = sorted(
            score_index, key=lambda tup: tup[0], reverse=True
        )
        sorted_score_index = sorted_score_index[:keep_top_k]
        selected_indices = {}

        for _, c, _ in sorted_score_index:
            selected_indices[c] = []
        for s, c, idx in sorted_score_index:
            selected_indices[c].append(idx)
        if not shared:
            for labels in selected_indices:
                selected_indices[labels].sort()
        num_det = keep_top_k

    return selected_indices, num_det


def batched_multiclass_nms(
    boxes,
    scores,
    background,
    score_threshold,
    nms_threshold,
    nms_top_k,
    keep_top_k,
    normalized=True,
):
    batch_size = scores.shape[0]
    num_boxes = scores.shape[2]
    det_outs = []

    lod = []
    for n in range(batch_size):
        nmsed_outs, nmsed_num = multiclass_nms(
            boxes[n],
            scores[n],
            background,
            score_threshold,
            nms_threshold,
            nms_top_k,
            keep_top_k,
            normalized,
            shared=True,
        )
        lod.append(nmsed_num)

        if nmsed_num == 0:
            continue
        tmp_det_out = []
        for c, indices in nmsed_outs.items():
            for idx in indices:
                xmin, ymin, xmax, ymax = boxes[n][idx][:]
                tmp_det_out.append(
                    [
                        c,
                        scores[n][c][idx],
                        xmin,
                        ymin,
                        xmax,
                        ymax,
                        idx + n * num_boxes,
                    ]
                )
        sorted_det_out = sorted(
            tmp_det_out, key=lambda tup: tup[0], reverse=False
        )
        det_outs.extend(sorted_det_out)
    return det_outs, lod


class TestLocalAwareNMSOp(OpTest):
    def set_argument(self):
        self.score_threshold = 0.01

    def setUp(self):
        self.set_argument()
        N = 10
        M = 1200
        C = 1
        BOX_SIZE = 4
        background = -1
        nms_threshold = 0.3
        nms_top_k = 400
        keep_top_k = 10
        score_threshold = self.score_threshold

        scores = np.random.random((N * M, C)).astype('float32')

        def softmax(x):
            # clip to shiftx, otherwise, when calc loss with
            # log(exp(shiftx)), may get log(0)=INF
            shiftx = (x - np.max(x)).clip(-64.0)
            exps = np.exp(shiftx)
            return exps / np.sum(exps)

        scores = np.apply_along_axis(softmax, 1, scores)
        scores = np.reshape(scores, (N, M, C))
        scores = np.transpose(scores, (0, 2, 1))

        boxes = np.random.random((N, M, BOX_SIZE)).astype('float32')
        boxes[:, :, 0:2] = boxes[:, :, 0:2] * 0.5
        boxes[:, :, 2:4] = boxes[:, :, 2:4] * 0.5 + 0.5

        boxes_copy = copy.deepcopy(boxes)
        scores_copy = copy.deepcopy(scores)
        det_outs, lod = batched_multiclass_nms(
            boxes_copy,
            scores_copy,
            background,
            score_threshold,
            nms_threshold,
            nms_top_k,
            keep_top_k,
        )

        lod = [1] if not det_outs else lod
        det_outs = [[-1, 0]] if not det_outs else det_outs
        det_outs = np.array(det_outs)
        nmsed_outs = det_outs[:, :-1].astype('float32')

        self.op_type = 'locality_aware_nms'
        self.inputs = {'BBoxes': boxes, 'Scores': scores}
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'background_label': background,
            'nms_threshold': nms_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'score_threshold': score_threshold,
            'nms_eta': 1.0,
            'normalized': True,
        }

    def test_check_output(self):
        self.check_output()


class TestLocalAwareNMSOpNoBoxes(TestLocalAwareNMSOp):
    def set_argument(self):
        self.score_threshold = 2.0


class TestLocalAwareNMSOp4Points(OpTest):
    def set_argument(self):
        self.score_threshold = 0.01

    def setUp(self):
        self.set_argument()
        N = 2
        M = 2
        C = 1
        BOX_SIZE = 8
        nms_top_k = 400
        keep_top_k = 200
        nms_threshold = 0.3
        score_threshold = self.score_threshold

        scores = np.array(
            [[[0.76319082, 0.73770091]], [[0.68513154, 0.45952697]]]
        )
        boxes = np.array(
            [
                [
                    [
                        0.42078365,
                        0.58117018,
                        2.92776169,
                        3.28557757,
                        4.24344318,
                        0.92196165,
                        2.72370856,
                        -1.66141214,
                    ],
                    [
                        0.13856006,
                        1.86871034,
                        2.81287224,
                        3.61381734,
                        4.5505249,
                        0.51766346,
                        2.75630304,
                        -1.91459389,
                    ],
                ],
                [
                    [
                        1.57533883,
                        1.3217477,
                        3.07904942,
                        3.89512545,
                        4.78680923,
                        1.96914586,
                        3.539482,
                        -1.59739244,
                    ],
                    [
                        0.55084125,
                        1.71596215,
                        2.52476074,
                        3.18940435,
                        5.09035159,
                        0.91959482,
                        3.71442385,
                        -0.57299128,
                    ],
                ],
            ]
        )

        det_outs = np.array(
            [
                [
                    0.0,
                    1.5008917,
                    0.28206837,
                    1.2140071,
                    2.8712926,
                    3.4469104,
                    4.3943763,
                    0.7232457,
                    2.7397292,
                    -1.7858533,
                ],
                [
                    0.0,
                    1.1446586,
                    1.1640508,
                    1.4800063,
                    2.856528,
                    3.6118112,
                    4.908667,
                    1.5478,
                    3.609713,
                    -1.1861432,
                ],
            ]
        )
        lod = [1, 1]
        nmsed_outs = det_outs.astype('float32')

        self.op_type = 'locality_aware_nms'
        self.inputs = {
            'BBoxes': boxes.astype('float32'),
            'Scores': scores.astype('float32'),
        }
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'score_threshold': score_threshold,
            'nms_threshold': nms_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'background_label': -1,
            'normalized': False,
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
