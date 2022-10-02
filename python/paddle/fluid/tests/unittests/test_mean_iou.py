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
import paddle.fluid as fluid
import paddle


def compute_mean_iou(predictions, labels, num_classes, in_wrongs, in_corrects,
                     in_mean_ious):
    assert predictions.shape == labels.shape
    predictions = predictions.flatten()
    labels = labels.flatten()

    out_wrong = np.zeros([num_classes]).astype("int32")
    for _, wrong in in_wrongs:
        out_wrong += wrong
    out_correct = np.zeros([num_classes]).astype("int32")
    for _, correct in in_corrects:
        out_correct += correct

    for pred, label in zip(predictions, labels):
        if pred == label:
            out_correct[pred] += 1
        else:
            out_wrong[pred] += 1
            out_wrong[label] += 1

    denominator = out_wrong + out_correct
    valid_count = (denominator != 0).sum()
    denominator = np.where(denominator > 0, denominator,
                           np.ones(denominator.shape))
    mean_iou = (out_correct / denominator).sum() / valid_count

    for _, in_mean_iou in in_mean_ious:
        mean_iou += in_mean_iou
    return mean_iou, out_wrong, out_correct


class TestMeanIOUOp(OpTest):

    def setUp(self):
        self.config()
        self.op_type = "mean_iou"
        predictions = np.random.randint(0, self.num_classes,
                                        self.image_size).astype("int32")
        labels = np.random.randint(0, self.num_classes,
                                   self.image_size).astype("int32")

        in_wrongs = []
        for i in range(self.in_wrong_num):
            in_wrongs.append(
                ("in_wrong_%d" % i,
                 np.random.randint(0, 10, [self.num_classes]).astype("int32")))

        in_corrects = []
        for i in range(self.in_correct_num):
            in_corrects.append(
                ("in_correct_%d" % i,
                 np.random.randint(0, 10, [self.num_classes]).astype("int32")))

        in_mean_ious = []
        for i in range(self.in_mean_iou_num):
            in_mean_ious.append(("in_mean_iou_%d" % i,
                                 np.random.uniform(0, 1,
                                                   [1]).astype("float32")))

        self.inputs = {
            'Predictions': predictions,
            'Labels': labels,
            'InWrongs': in_wrongs,
            'InCorrects': in_corrects,
            'InMeanIou': in_mean_ious
        }
        self.attrs = {'num_classes': int(self.num_classes)}
        mean_iou, out_wrong, out_correct = compute_mean_iou(
            predictions, labels, self.num_classes, in_wrongs, in_corrects,
            in_mean_ious)
        self.outputs = {
            'OutMeanIou': mean_iou,
            'OutWrong': out_wrong,
            'OutCorrect': out_correct
        }

    def config(self):
        self.num_classes = 10
        self.image_size = [128, 128]
        self.in_wrong_num = 0
        self.in_correct_num = 0
        self.in_mean_iou_num = 0

    def test_check_output(self):
        self.check_output()


class TestCase1(TestMeanIOUOp):

    def config(self):
        self.num_classes = 5
        self.image_size = [100, 128]
        self.in_wrong_num = 2
        self.in_correct_num = 2
        self.in_mean_iou_num = 2

    # NOTE(dev): Skip check_dygraph becuase Python API doesn't expose
    # in_wrong_num/in_correct_num/in_mean_iou_num argument
    def test_check_output(self):
        self.check_output(check_dygraph=False, check_eager=False)


class TestMeanIOUOpError(unittest.TestCase):

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            # The input type of accuracy_op must be Variable.
            x1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            y1 = fluid.create_lod_tensor(np.array([[-1]]), [[1]],
                                         fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.mean_iou, x1, y1)
            # The input dtype of accuracy_op must be float32 or float64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="float32")
            y2 = fluid.layers.data(name='x2', shape=[4], dtype="float32")
            self.assertRaises(TypeError, fluid.layers.mean_iou, x2, y2)


if __name__ == '__main__':
    unittest.main()
