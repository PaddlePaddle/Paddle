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
import six
import sys
import collections
import math
import paddle.fluid as fluid
from op_test import OpTest


class TestDetectionMAPOp(OpTest):

    def set_data(self):
        self.class_num = 4
        self.init_test_case()
        self.mAP = [self.calc_map(self.tf_pos, self.tf_pos_lod)]
        self.label = np.array(self.label).astype('float32')
        self.detect = np.array(self.detect).astype('float32')
        self.mAP = np.array(self.mAP).astype('float32')

        if len(self.class_pos_count) > 0:
            self.class_pos_count = np.array(
                self.class_pos_count).astype('int32')
            self.true_pos = np.array(self.true_pos).astype('float32')
            self.false_pos = np.array(self.false_pos).astype('float32')
            self.has_state = np.array([1]).astype('int32')

            self.inputs = {
                'Label': (self.label, self.label_lod),
                'DetectRes': (self.detect, self.detect_lod),
                'HasState': self.has_state,
                'PosCount': self.class_pos_count,
                'TruePos': (self.true_pos, self.true_pos_lod),
                'FalsePos': (self.false_pos, self.false_pos_lod)
            }
        else:
            self.inputs = {
                'Label': (self.label, self.label_lod),
                'DetectRes': (self.detect, self.detect_lod),
            }

        self.attrs = {
            'overlap_threshold': self.overlap_threshold,
            'evaluate_difficult': self.evaluate_difficult,
            'ap_type': self.ap_type,
            'class_num': self.class_num
        }

        self.out_class_pos_count = np.array(
            self.out_class_pos_count).astype('int')
        self.out_true_pos = np.array(self.out_true_pos).astype('float32')
        self.out_false_pos = np.array(self.out_false_pos).astype('float32')

        self.outputs = {
            'MAP': self.mAP,
            'AccumPosCount': self.out_class_pos_count,
            'AccumTruePos': (self.out_true_pos, self.out_true_pos_lod),
            'AccumFalsePos': (self.out_false_pos, self.out_false_pos_lod)
        }

    def init_test_case(self):
        self.overlap_threshold = 0.3
        self.evaluate_difficult = True
        self.ap_type = "integral"

        self.label_lod = [[2, 2]]
        # label difficult xmin ymin xmax ymax
        self.label = [[1, 0, 0.1, 0.1, 0.3, 0.3], [1, 1, 0.6, 0.6, 0.8, 0.8],
                      [2, 0, 0.3, 0.3, 0.6, 0.5], [1, 0, 0.7, 0.1, 0.9, 0.3]]

        # label score xmin ymin xmax ymax difficult
        self.detect_lod = [[3, 4]]
        self.detect = [[1, 0.3, 0.1, 0.0, 0.4,
                        0.3], [1, 0.7, 0.0, 0.1, 0.2, 0.3],
                       [1, 0.9, 0.7, 0.6, 0.8,
                        0.8], [2, 0.8, 0.2, 0.1, 0.4, 0.4],
                       [2, 0.1, 0.4, 0.3, 0.7,
                        0.5], [1, 0.2, 0.8, 0.1, 1.0, 0.3],
                       [3, 0.2, 0.8, 0.1, 1.0, 0.3]]

        # label score true_pos false_pos
        self.tf_pos_lod = [[3, 4]]
        self.tf_pos = [[1, 0.9, 1, 0], [1, 0.7, 1, 0], [1, 0.3, 0, 1],
                       [1, 0.2, 1, 0], [2, 0.8, 0, 1], [2, 0.1, 1, 0],
                       [3, 0.2, 0, 1]]

        self.class_pos_count = []
        self.true_pos_lod = [[]]
        self.true_pos = [[]]
        self.false_pos_lod = [[]]
        self.false_pos = [[]]

    def calc_map(self, tf_pos, tf_pos_lod):
        mAP = 0.0
        count = 0

        def get_input_pos(class_pos_count, true_pos, true_pos_lod, false_pos,
                          false_pos_lod):
            class_pos_count_dict = collections.Counter()
            true_pos_dict = collections.defaultdict(list)
            false_pos_dict = collections.defaultdict(list)
            for i, count in enumerate(class_pos_count):
                class_pos_count_dict[i] = count

            cur_pos = 0
            for i in range(len(true_pos_lod[0])):
                start = cur_pos
                cur_pos += true_pos_lod[0][i]
                end = cur_pos
                for j in range(start, end):
                    true_pos_dict[i].append(true_pos[j])

            cur_pos = 0
            for i in range(len(false_pos_lod[0])):
                start = cur_pos
                cur_pos += false_pos_lod[0][i]
                end = cur_pos
                for j in range(start, end):
                    false_pos_dict[i].append(false_pos[j])

            return class_pos_count_dict, true_pos_dict, false_pos_dict

        def get_output_pos(label_count, true_pos, false_pos):
            label_number = self.class_num

            out_class_pos_count = []
            out_true_pos_lod = []
            out_true_pos = []
            out_false_pos_lod = []
            out_false_pos = []

            for i in range(label_number):
                out_class_pos_count.append([label_count[i]])
                true_pos_list = true_pos[i]
                out_true_pos += true_pos_list
                out_true_pos_lod.append(len(true_pos_list))
                false_pos_list = false_pos[i]
                out_false_pos += false_pos_list
                out_false_pos_lod.append(len(false_pos_list))

            return out_class_pos_count, out_true_pos, [
                out_true_pos_lod
            ], out_false_pos, [out_false_pos_lod]

        def get_accumulation(pos_list):
            sorted_list = sorted(pos_list, key=lambda pos: pos[0], reverse=True)
            sum = 0
            accu_list = []
            for (score, count) in sorted_list:
                sum += count
                accu_list.append(sum)
            return accu_list

        label_count, true_pos, false_pos = get_input_pos(
            self.class_pos_count, self.true_pos, self.true_pos_lod,
            self.false_pos, self.false_pos_lod)
        for v in self.label:
            label = v[0]
            difficult = False if len(v) == 5 else v[1]
            if self.evaluate_difficult:
                label_count[label] += 1
            elif not difficult:
                label_count[label] += 1

        for (label, score, tp, fp) in tf_pos:
            true_pos[label].append([score, tp])
            false_pos[label].append([score, fp])

        for (label, label_pos_num) in six.iteritems(label_count):
            if label_pos_num == 0: continue
            if label not in true_pos:
                count += 1
                continue
            label_true_pos = true_pos[label]
            label_false_pos = false_pos[label]

            accu_tp_sum = get_accumulation(label_true_pos)
            accu_fp_sum = get_accumulation(label_false_pos)

            precision = []
            recall = []

            for i in range(len(accu_tp_sum)):
                precision.append(
                    float(accu_tp_sum[i]) /
                    float(accu_tp_sum[i] + accu_fp_sum[i]))
                recall.append(float(accu_tp_sum[i]) / label_pos_num)

            if self.ap_type == "11point":
                max_precisions = [0.0] * 11
                start_idx = len(accu_tp_sum) - 1
                for j in range(10, -1, -1):
                    for i in range(start_idx, -1, -1):
                        if recall[i] < float(j) / 10.0:
                            start_idx = i
                            if j > 0:
                                max_precisions[j - 1] = max_precisions[j]
                                break
                        else:
                            if max_precisions[j] < precision[i]:
                                max_precisions[j] = precision[i]
                for j in range(10, -1, -1):
                    mAP += max_precisions[j] / 11
                count += 1
            elif self.ap_type == "integral":
                average_precisions = 0.0
                prev_recall = 0.0
                for i in range(len(accu_tp_sum)):
                    if math.fabs(recall[i] - prev_recall) > 1e-6:
                        average_precisions += precision[i] * \
                            math.fabs(recall[i] - prev_recall)
                        prev_recall = recall[i]

                mAP += average_precisions
                count += 1
        pcnt, tp, tp_lod, fp, fp_lod = get_output_pos(label_count, true_pos,
                                                      false_pos)
        self.out_class_pos_count = pcnt
        self.out_true_pos = tp
        self.out_true_pos_lod = tp_lod
        self.out_false_pos = fp
        self.out_false_pos_lod = fp_lod
        if count != 0:
            mAP /= count
        return mAP

    def setUp(self):
        self.op_type = "detection_map"
        self.set_data()

    def test_check_output(self):
        self.check_output()


class TestDetectionMAPOpSkipDiff(TestDetectionMAPOp):

    def init_test_case(self):
        super(TestDetectionMAPOpSkipDiff, self).init_test_case()

        self.evaluate_difficult = False

        self.tf_pos_lod = [[2, 4]]
        # label score true_pos false_pos
        self.tf_pos = [[1, 0.7, 1, 0], [1, 0.3, 0, 1], [1, 0.2, 1, 0],
                       [2, 0.8, 0, 1], [2, 0.1, 1, 0], [3, 0.2, 0, 1]]


class TestDetectionMAPOpWithoutDiff(TestDetectionMAPOp):

    def init_test_case(self):
        super(TestDetectionMAPOpWithoutDiff, self).init_test_case()

        # label xmin ymin xmax ymax
        self.label = [[1, 0.1, 0.1, 0.3, 0.3], [1, 0.6, 0.6, 0.8, 0.8],
                      [2, 0.3, 0.3, 0.6, 0.5], [1, 0.7, 0.1, 0.9, 0.3]]


class TestDetectionMAPOp11Point(TestDetectionMAPOp):

    def init_test_case(self):
        super(TestDetectionMAPOp11Point, self).init_test_case()

        self.ap_type = "11point"


class TestDetectionMAPOpMultiBatch(TestDetectionMAPOp):

    def init_test_case(self):
        super(TestDetectionMAPOpMultiBatch, self).init_test_case()
        self.class_pos_count = [0, 2, 1, 0]
        self.true_pos_lod = [[0, 3, 2]]
        self.true_pos = [[0.7, 1.], [0.3, 0.], [0.2, 1.], [0.8, 0.], [0.1, 1.]]
        self.false_pos_lod = [[0, 3, 2]]
        self.false_pos = [[0.7, 0.], [0.3, 1.], [0.2, 0.], [0.8, 1.], [0.1, 0.]]


class TestDetectionMAPOp11PointWithClassNoTP(TestDetectionMAPOp):

    def init_test_case(self):
        self.overlap_threshold = 0.3
        self.evaluate_difficult = True
        self.ap_type = "11point"

        self.label_lod = [[2]]
        # label difficult xmin ymin xmax ymax
        self.label = [[2, 0, 0.3, 0.3, 0.6, 0.5], [1, 0, 0.7, 0.1, 0.9, 0.3]]

        # label score xmin ymin xmax ymax difficult
        self.detect_lod = [[1]]
        self.detect = [[1, 0.2, 0.8, 0.1, 1.0, 0.3]]

        # label score true_pos false_pos
        self.tf_pos_lod = [[3, 4]]
        self.tf_pos = [[1, 0.2, 1, 0]]

        self.class_pos_count = []
        self.true_pos_lod = [[]]
        self.true_pos = [[]]
        self.false_pos_lod = [[]]
        self.false_pos = [[]]


if __name__ == '__main__':
    unittest.main()
