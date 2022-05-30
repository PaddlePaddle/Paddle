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
from test_anchor_generator_op import anchor_generator_in_python
from test_multiclass_nms_op import iou
from test_multiclass_nms_op import nms
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


def multiclass_nms(prediction, class_num, keep_top_k, nms_threshold):
    selected_indices = {}
    num_det = 0
    for c in range(class_num):
        if c not in prediction.keys():
            continue
        cls_dets = prediction[c]
        all_scores = np.zeros(len(cls_dets))
        for i in range(all_scores.shape[0]):
            all_scores[i] = cls_dets[i][4]
        indices = nms(cls_dets, all_scores, 0.0, nms_threshold, -1, False, 1.0)
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
        nmsed_outs.append([c + 1, s, xmin, ymin, xmax, ymax])

    return nmsed_outs, num_det


def retinanet_detection_out(boxes_list, scores_list, anchors_list, im_info,
                            score_threshold, nms_threshold, nms_top_k,
                            keep_top_k):
    class_num = scores_list[0].shape[-1]
    im_height, im_width, im_scale = im_info

    num_level = len(scores_list)
    prediction = {}
    for lvl in range(num_level):
        scores_per_level = scores_list[lvl]
        scores_per_level = scores_per_level.flatten()
        bboxes_per_level = boxes_list[lvl]
        bboxes_per_level = bboxes_per_level.flatten()
        anchors_per_level = anchors_list[lvl]
        anchors_per_level = anchors_per_level.flatten()

        thresh = score_threshold if lvl < (num_level - 1) else 0.0
        selected_indices = np.argwhere(scores_per_level > thresh)
        scores = scores_per_level[selected_indices]
        sorted_indices = np.argsort(-scores, axis=0, kind='mergesort')
        if nms_top_k > -1 and nms_top_k < sorted_indices.shape[0]:
            sorted_indices = sorted_indices[:nms_top_k]

        for i in range(sorted_indices.shape[0]):
            idx = selected_indices[sorted_indices[i]]
            idx = idx[0][0]
            a = int(idx / class_num)
            c = int(idx % class_num)
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


def batched_retinanet_detection_out(boxes, scores, anchors, im_info,
                                    score_threshold, nms_threshold, nms_top_k,
                                    keep_top_k):
    batch_size = scores[0].shape[0]
    det_outs = []
    lod = []

    for n in range(batch_size):
        boxes_per_batch = []
        scores_per_batch = []

        num_level = len(scores)
        for lvl in range(num_level):
            boxes_per_batch.append(boxes[lvl][n])
            scores_per_batch.append(scores[lvl][n])

        nmsed_outs, nmsed_num = retinanet_detection_out(
            boxes_per_batch, scores_per_batch, anchors, im_info[n],
            score_threshold, nms_threshold, nms_top_k, keep_top_k)
        lod.append(nmsed_num)
        if nmsed_num == 0:
            continue

        det_outs.extend(nmsed_outs)
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

        self.layer_h = []
        self.layer_w = []
        num_levels = self.max_level - self.min_level + 1
        for i in range(num_levels):
            self.layer_h.append(2**(num_levels - i))
            self.layer_w.append(2**(num_levels - i))

    def init_test_input(self):
        anchor_num = len(self.aspect_ratios) * self.scales_per_octave
        num_levels = self.max_level - self.min_level + 1
        self.scores_list = []
        self.bboxes_list = []
        self.anchors_list = []

        for i in range(num_levels):
            layer_h = self.layer_h[i]
            layer_w = self.layer_w[i]

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
            self.scores_list.append(score.astype('float32'))
            self.bboxes_list.append(box.astype('float32'))
            self.anchors_list.append(anchor.astype('float32'))

        self.im_info = np.array([[256., 256., 1.5]]).astype(
            'float32')  #im_height, im_width, scale

    def setUp(self):
        self.set_argument()
        self.init_test_input()

        nmsed_outs, lod = batched_retinanet_detection_out(
            self.bboxes_list, self.scores_list, self.anchors_list, self.im_info,
            self.score_threshold, self.nms_threshold, self.nms_top_k,
            self.keep_top_k)
        nmsed_outs = np.array(nmsed_outs).astype('float32')
        self.op_type = 'retinanet_detection_output'
        self.inputs = {
            'BBoxes': [('b0', self.bboxes_list[0]), ('b1', self.bboxes_list[1]),
                       ('b2', self.bboxes_list[2]), ('b3', self.bboxes_list[3]),
                       ('b4', self.bboxes_list[4])],
            'Scores': [('s0', self.scores_list[0]), ('s1', self.scores_list[1]),
                       ('s2', self.scores_list[2]), ('s3', self.scores_list[3]),
                       ('s4', self.scores_list[4])],
            'Anchors':
            [('a0', self.anchors_list[0]), ('a1', self.anchors_list[1]),
             ('a2', self.anchors_list[2]), ('a3', self.anchors_list[3]),
             ('a4', self.anchors_list[4])],
            'ImInfo': (self.im_info, [[1, ]])
        }
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'score_threshold': self.score_threshold,
            'nms_top_k': self.nms_top_k,
            'nms_threshold': self.nms_threshold,
            'keep_top_k': self.keep_top_k,
            'nms_eta': 1.,
        }

    def test_check_output(self):
        self.check_output()


class TestRetinanetDetectionOutOp2(OpTest):
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
        # Here test the case there the shape of each FPN level
        # is irrelevant.
        self.layer_h = [1, 4, 8, 8, 16]
        self.layer_w = [1, 4, 8, 8, 16]


class TestRetinanetDetectionOutOpNo3(TestRetinanetDetectionOutOp1):
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

        self.layer_h = []
        self.layer_w = []
        num_levels = self.max_level - self.min_level + 1
        for i in range(num_levels):
            self.layer_h.append(2**(num_levels - i))
            self.layer_w.append(2**(num_levels - i))


class TestRetinanetDetectionOutOpNo4(TestRetinanetDetectionOutOp1):
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

        self.layer_h = []
        self.layer_w = []
        num_levels = self.max_level - self.min_level + 1
        for i in range(num_levels):
            self.layer_h.append(2**(num_levels - i))
            self.layer_w.append(2**(num_levels - i))

    def setUp(self):
        self.set_argument()
        self.init_test_input()

        nmsed_outs, lod = batched_retinanet_detection_out(
            self.bboxes_list, self.scores_list, self.anchors_list, self.im_info,
            self.score_threshold, self.nms_threshold, self.nms_top_k,
            self.keep_top_k)
        nmsed_outs = np.array(nmsed_outs).astype('float32')
        self.op_type = 'retinanet_detection_output'
        self.inputs = {
            'BBoxes':
            [('b0', self.bboxes_list[0]), ('b1', self.bboxes_list[1]),
             ('b2', self.bboxes_list[2]), ('b3', self.bboxes_list[3])],
            'Scores': [('s0', self.scores_list[0]), ('s1', self.scores_list[1]),
                       ('s2', self.scores_list[2]),
                       ('s3', self.scores_list[3])],
            'Anchors':
            [('a0', self.anchors_list[0]), ('a1', self.anchors_list[1]),
             ('a2', self.anchors_list[2]), ('a3', self.anchors_list[3])],
            'ImInfo': (self.im_info, [[1, ]])
        }
        self.outputs = {'Out': (nmsed_outs, [lod])}
        self.attrs = {
            'score_threshold': self.score_threshold,
            'nms_top_k': self.nms_top_k,
            'nms_threshold': self.nms_threshold,
            'keep_top_k': self.keep_top_k,
            'nms_eta': 1.,
        }

    def test_check_output(self):
        self.check_output()


class TestRetinanetDetectionOutOpNo5(TestRetinanetDetectionOutOp1):
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

        self.layer_h = []
        self.layer_w = []
        num_levels = self.max_level - self.min_level + 1
        for i in range(num_levels):
            self.layer_h.append(2**(num_levels - i))
            self.layer_w.append(2**(num_levels - i))


class TestRetinanetDetectionOutOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            bboxes_low1 = fluid.data(
                name='bboxes_low1', shape=[1, 44, 4], dtype='float32')
            bboxes_high1 = fluid.data(
                name='bboxes_high1', shape=[1, 11, 4], dtype='float32')
            scores_low1 = fluid.data(
                name='scores_low1', shape=[1, 44, 10], dtype='float32')
            scores_high1 = fluid.data(
                name='scores_high1', shape=[1, 11, 10], dtype='float32')
            anchors_low1 = fluid.data(
                name='anchors_low1', shape=[44, 4], dtype='float32')
            anchors_high1 = fluid.data(
                name='anchors_high1', shape=[11, 4], dtype='float32')
            im_info1 = fluid.data(
                name="im_info1", shape=[1, 3], dtype='float32')

            # The `bboxes` must be list, each element must be Variable and 
            # its Tensor data type must be one of float32 and float64.
            def test_bboxes_type():
                fluid.layers.retinanet_detection_output(
                    bboxes=bboxes_low1,
                    scores=[scores_low1, scores_high1],
                    anchors=[anchors_low1, anchors_high1],
                    im_info=im_info1)

            self.assertRaises(TypeError, test_bboxes_type)

            def test_bboxes_tensor_dtype():
                bboxes_high2 = fluid.data(
                    name='bboxes_high2', shape=[1, 11, 4], dtype='int32')
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_high2, 5],
                    scores=[scores_low1, scores_high1],
                    anchors=[anchors_low1, anchors_high1],
                    im_info=im_info1)

            self.assertRaises(TypeError, test_bboxes_tensor_dtype)

            # The `scores` must be list, each element must be Variable and its
            # Tensor data type must be one of float32 and float64.
            def test_scores_type():
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=scores_low1,
                    anchors=[anchors_low1, anchors_high1],
                    im_info=im_info1)

            self.assertRaises(TypeError, test_scores_type)

            def test_scores_tensor_dtype():
                scores_high2 = fluid.data(
                    name='scores_high2', shape=[1, 11, 10], dtype='int32')
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=[scores_high2, 5],
                    anchors=[anchors_low1, anchors_high1],
                    im_info=im_info1)

            self.assertRaises(TypeError, test_scores_tensor_dtype)

            # The `anchors` must be list, each element must be Variable and its
            # Tensor data type must be one of float32 and float64.
            def test_anchors_type():
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=[scores_low1, scores_high1],
                    anchors=anchors_low1,
                    im_info=im_info1)

            self.assertRaises(TypeError, test_anchors_type)

            def test_anchors_tensor_dtype():
                anchors_high2 = fluid.data(
                    name='anchors_high2', shape=[11, 4], dtype='int32')
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=[scores_low1, scores_high1],
                    anchors=[anchors_high2, 5],
                    im_info=im_info1)

            self.assertRaises(TypeError, test_anchors_tensor_dtype)

            # The `im_info` must be Variable and the data type of `im_info`
            # Tensor must be one of float32 and float64.
            def test_iminfo_type():
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=[scores_low1, scores_high1],
                    anchors=[anchors_low1, anchors_high1],
                    im_info=[2, 3, 4])

            self.assertRaises(TypeError, test_iminfo_type)

            def test_iminfo_tensor_dtype():
                im_info2 = fluid.data(
                    name='im_info2', shape=[1, 3], dtype='int32')
                fluid.layers.retinanet_detection_output(
                    bboxes=[bboxes_low1, bboxes_high1],
                    scores=[scores_low1, scores_high1],
                    anchors=[anchors_low1, anchors_high1],
                    im_info=im_info2)

            self.assertRaises(TypeError, test_iminfo_tensor_dtype)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
