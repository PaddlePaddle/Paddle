#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import unittest

import numpy as np
from op_test import OpTest

import paddle


def python_matrix_nms(
    bboxes,
    scores,
    score_threshold,
    nms_top_k,
    keep_top_k,
    post_threshold,
    use_gaussian=False,
    gaussian_sigma=2.0,
    background_label=0,
    normalized=True,
    return_index=True,
    return_rois_num=True,
):
    out, rois_num, index = paddle.vision.ops.matrix_nms(
        bboxes,
        scores,
        score_threshold,
        post_threshold,
        nms_top_k,
        keep_top_k,
        use_gaussian,
        gaussian_sigma,
        background_label,
        normalized,
        return_index,
        return_rois_num,
    )
    if not return_index:
        index = None
    if not return_rois_num:
        rois_num = None
    return out, index, rois_num


def softmax(x):
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.0)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def iou_matrix(a, b, norm=True):
    tl_i = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    br_i = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    pad = not norm and 1 or 0

    area_i = np.prod(br_i - tl_i + pad, axis=2) * (tl_i < br_i).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2] + pad, axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2] + pad, axis=1)
    area_o = area_a[:, np.newaxis] + area_b - area_i
    return area_i / (area_o + 1e-10)


def matrix_nms(
    boxes,
    scores,
    score_threshold,
    post_threshold=0.0,
    nms_top_k=400,
    normalized=True,
    use_gaussian=False,
    gaussian_sigma=2.0,
):
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()
    selected_indices = np.where(all_scores > score_threshold)[0]
    all_scores = all_scores[selected_indices]

    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')
    sorted_scores = all_scores[sorted_indices]
    sorted_indices = selected_indices[sorted_indices]
    if nms_top_k > -1 and nms_top_k < sorted_indices.shape[0]:
        sorted_indices = sorted_indices[:nms_top_k]
        sorted_scores = sorted_scores[:nms_top_k]

    selected_boxes = boxes[sorted_indices, :]
    ious = iou_matrix(selected_boxes, selected_boxes)
    ious = np.triu(ious, k=1)
    iou_cmax = ious.max(0)
    N = iou_cmax.shape[0]
    iou_cmax = np.repeat(iou_cmax[:, np.newaxis], N, axis=1)

    if use_gaussian:
        decay = np.exp((iou_cmax**2 - ious**2) * gaussian_sigma)
    else:
        decay = (1 - ious) / (1 - iou_cmax)
    decay = decay.min(0)
    decayed_scores = sorted_scores * decay

    if post_threshold > 0.0:
        inds = np.where(decayed_scores > post_threshold)[0]
        selected_boxes = selected_boxes[inds, :]
        decayed_scores = decayed_scores[inds]
        sorted_indices = sorted_indices[inds]

    return decayed_scores, selected_boxes, sorted_indices


def multiclass_nms(
    boxes,
    scores,
    background,
    score_threshold,
    post_threshold,
    nms_top_k,
    keep_top_k,
    normalized,
    use_gaussian,
    gaussian_sigma,
):
    all_boxes = []
    all_cls = []
    all_scores = []
    all_indices = []
    for c in range(scores.shape[0]):
        if c == background:
            continue
        decayed_scores, selected_boxes, indices = matrix_nms(
            boxes,
            scores[c],
            score_threshold,
            post_threshold,
            nms_top_k,
            normalized,
            use_gaussian,
            gaussian_sigma,
        )
        all_cls.append(np.full(len(decayed_scores), c, decayed_scores.dtype))
        all_boxes.append(selected_boxes)
        all_scores.append(decayed_scores)
        all_indices.append(indices)

    all_cls = np.concatenate(all_cls)
    all_boxes = np.concatenate(all_boxes)
    all_scores = np.concatenate(all_scores)
    all_indices = np.concatenate(all_indices)
    all_pred = np.concatenate(
        (all_cls[:, np.newaxis], all_scores[:, np.newaxis], all_boxes), axis=1
    )

    num_det = len(all_pred)
    if num_det == 0:
        return all_pred, np.array([], dtype=np.float32)

    inds = np.argsort(-all_scores, axis=0, kind='mergesort')
    all_pred = all_pred[inds, :]
    all_indices = all_indices[inds]

    if keep_top_k > -1 and num_det > keep_top_k:
        num_det = keep_top_k
        all_pred = all_pred[:keep_top_k, :]
        all_indices = all_indices[:keep_top_k]

    return all_pred, all_indices


def batched_multiclass_nms(
    boxes,
    scores,
    background,
    score_threshold,
    post_threshold,
    nms_top_k,
    keep_top_k,
    normalized=True,
    use_gaussian=False,
    gaussian_sigma=2.0,
):
    batch_size = scores.shape[0]
    det_outs = []
    index_outs = []
    lod = []
    for n in range(batch_size):
        nmsed_outs, indices = multiclass_nms(
            boxes[n],
            scores[n],
            background,
            score_threshold,
            post_threshold,
            nms_top_k,
            keep_top_k,
            normalized,
            use_gaussian,
            gaussian_sigma,
        )
        nmsed_num = len(nmsed_outs)
        lod.append(nmsed_num)
        if nmsed_num == 0:
            continue
        indices += n * scores.shape[2]
        det_outs.append(nmsed_outs)
        index_outs.append(indices)
    if det_outs:
        det_outs = np.concatenate(det_outs)
        index_outs = np.concatenate(index_outs)
    return det_outs, index_outs, lod


class TestMatrixNMSOp(OpTest):
    def set_argument(self):
        self.post_threshold = 0.0
        self.use_gaussian = False

    def setUp(self):
        self.set_argument()
        self.python_api = python_matrix_nms
        N = 7
        M = 1200
        C = 21
        BOX_SIZE = 4
        background = 0
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01
        post_threshold = self.post_threshold
        use_gaussian = False
        if hasattr(self, 'use_gaussian'):
            use_gaussian = self.use_gaussian
        gaussian_sigma = 2.0

        scores = np.random.random((N * M, C)).astype('float32')

        scores = np.apply_along_axis(softmax, 1, scores)
        scores = np.reshape(scores, (N, M, C))
        scores = np.transpose(scores, (0, 2, 1))

        boxes = np.random.random((N, M, BOX_SIZE)).astype('float32')
        boxes[:, :, 0:2] = boxes[:, :, 0:2] * 0.5
        boxes[:, :, 2:4] = boxes[:, :, 2:4] * 0.5 + 0.5

        det_outs, index_outs, lod = batched_multiclass_nms(
            boxes,
            scores,
            background,
            score_threshold,
            post_threshold,
            nms_top_k,
            keep_top_k,
            True,
            use_gaussian,
            gaussian_sigma,
        )

        empty = len(det_outs) == 0
        det_outs = (
            np.array([], dtype=np.float32).reshape([0, BOX_SIZE + 2])
            if empty
            else det_outs
        )
        index_outs = np.array([], dtype=np.float32) if empty else index_outs
        nmsed_outs = det_outs.astype('float32')

        self.op_type = 'matrix_nms'
        self.inputs = {'BBoxes': boxes, 'Scores': scores}
        self.outputs = {
            'Out': nmsed_outs,
            'Index': index_outs[:, None],
            'RoisNum': np.array(lod).astype('int32'),
        }
        self.attrs = {
            'score_threshold': score_threshold,
            'nms_top_k': nms_top_k,
            'keep_top_k': keep_top_k,
            'post_threshold': post_threshold,
            'use_gaussian': use_gaussian,
            'gaussian_sigma': gaussian_sigma,
            'background_label': 0,
            'normalized': True,
        }

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestMatrixNMSOpNoOutput(TestMatrixNMSOp):
    def set_argument(self):
        self.post_threshold = 2.0


class TestMatrixNMSOpGaussian(TestMatrixNMSOp):
    def set_argument(self):
        self.post_threshold = 0.0
        self.use_gaussian = True


class TestMatrixNMSError(unittest.TestCase):

    def test_errors(self):
        M = 1200
        N = 7
        C = 21
        BOX_SIZE = 4
        nms_top_k = 400
        keep_top_k = 200
        score_threshold = 0.01
        post_threshold = 0.0

        boxes_np = np.random.random((M, C, BOX_SIZE)).astype('float32')
        scores = np.random.random((N * M, C)).astype('float32')
        scores = np.apply_along_axis(softmax, 1, scores)
        scores = np.reshape(scores, (N, M, C))
        scores_np = np.transpose(scores, (0, 2, 1))

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            boxes_data = paddle.static.data(
                name='bboxes', shape=[M, C, BOX_SIZE], dtype='float32'
            )
            scores_data = paddle.static.data(
                name='scores', shape=[N, C, M], dtype='float32'
            )

            def test_bboxes_Variable():
                # the bboxes type must be Variable
                paddle.vision.ops.matrix_nms(
                    bboxes=boxes_np,
                    scores=scores_data,
                    score_threshold=score_threshold,
                    post_threshold=post_threshold,
                    nms_top_k=nms_top_k,
                    keep_top_k=keep_top_k,
                )

            def test_scores_Variable():
                # the scores type must be Variable
                paddle.vision.ops.matrix_nms(
                    bboxes=boxes_data,
                    scores=scores_np,
                    score_threshold=score_threshold,
                    post_threshold=post_threshold,
                    nms_top_k=nms_top_k,
                    keep_top_k=keep_top_k,
                )

            def test_empty():
                # when all score are lower than threshold
                try:
                    paddle.vision.ops.matrix_nms(
                        bboxes=boxes_data,
                        scores=scores_data,
                        score_threshold=score_threshold,
                        post_threshold=post_threshold,
                        nms_top_k=nms_top_k,
                        keep_top_k=keep_top_k,
                    )
                except Exception as e:
                    self.fail(e)

            def test_coverage():
                # cover correct workflow
                try:
                    paddle.vision.ops.matrix_nms(
                        bboxes=boxes_data,
                        scores=scores_data,
                        score_threshold=score_threshold,
                        post_threshold=post_threshold,
                        nms_top_k=nms_top_k,
                        keep_top_k=keep_top_k,
                    )
                except Exception as e:
                    self.fail(e)

            self.assertRaises(TypeError, test_bboxes_Variable)
            self.assertRaises(TypeError, test_scores_Variable)
            test_coverage()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
