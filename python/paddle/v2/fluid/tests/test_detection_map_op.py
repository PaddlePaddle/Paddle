import unittest
import numpy as np
import sys
import collections
import math
from op_test import OpTest


class TestDetectionMAPOp(OpTest):
    def set_data(self):
        self.init_test_case()

        self.mAP = [self.calc_map(self.tf_pos, self.tf_pos_lod)]
        self.label = np.array(self.label).astype('float32')
        self.detect = np.array(self.detect).astype('float32')
        self.mAP = np.array(self.mAP).astype('float32')

        self.inputs = {
            'Label': (self.label, self.label_lod),
            'Detection': (self.detect, self.detect_lod)
        }

        self.attrs = {
            'overlap_threshold': self.overlap_threshold,
            'evaluate_difficult': self.evaluate_difficult,
            'ap_type': self.ap_type
        }

        self.outputs = {'MAP': self.mAP}

    def init_test_case(self):
        self.overlap_threshold = 0.3
        self.evaluate_difficult = True
        self.ap_type = "integral"

        self.label_lod = [[0, 2, 4]]
        # label difficult xmin ymin xmax ymax
        self.label = [[1, 0, 0.1, 0.1, 0.3, 0.3], [1, 1, 0.6, 0.6, 0.8, 0.8],
                      [2, 0, 0.3, 0.3, 0.6, 0.5], [1, 0, 0.7, 0.1, 0.9, 0.3]]

        # label score xmin ymin xmax ymax difficult
        self.detect_lod = [[0, 3, 7]]
        self.detect = [
            [1, 0.3, 0.1, 0.0, 0.4, 0.3], [1, 0.7, 0.0, 0.1, 0.2, 0.3],
            [1, 0.9, 0.7, 0.6, 0.8, 0.8], [2, 0.8, 0.2, 0.1, 0.4, 0.4],
            [2, 0.1, 0.4, 0.3, 0.7, 0.5], [1, 0.2, 0.8, 0.1, 1.0, 0.3],
            [3, 0.2, 0.8, 0.1, 1.0, 0.3]
        ]

        # label score true_pos false_pos
        self.tf_pos_lod = [[0, 3, 7]]
        self.tf_pos = [[1, 0.9, 1, 0], [1, 0.7, 1, 0], [1, 0.3, 0, 1],
                       [1, 0.2, 1, 0], [2, 0.8, 0, 1], [2, 0.1, 1, 0],
                       [3, 0.2, 0, 1]]

    def calc_map(self, tf_pos, tf_pos_lod):
        mAP = 0.0
        count = 0

        class_pos_count = {}
        true_pos = {}
        false_pos = {}

        def get_accumulation(pos_list):
            sorted_list = sorted(pos_list, key=lambda pos: pos[0], reverse=True)
            sum = 0
            accu_list = []
            for (score, count) in sorted_list:
                sum += count
                accu_list.append(sum)
            return accu_list

        label_count = collections.Counter()
        for (label, difficult, xmin, ymin, xmax, ymax) in self.label:
            if self.evaluate_difficult:
                label_count[label] += 1
            elif not difficult:
                label_count[label] += 1

        true_pos = collections.defaultdict(list)
        false_pos = collections.defaultdict(list)
        for (label, score, tp, fp) in tf_pos:
            true_pos[label].append([score, tp])
            false_pos[label].append([score, fp])

        for (label, label_pos_num) in label_count.items():
            if label_pos_num == 0 or label not in true_pos:
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

        if count != 0: mAP /= count
        return mAP * 100.0

    def setUp(self):
        self.op_type = "detection_map"
        self.set_data()

    def test_check_output(self):
        self.check_output()


class TestDetectionMAPOpSkipDiff(TestDetectionMAPOp):
    def init_test_case(self):
        super(TestDetectionMAPOpSkipDiff, self).init_test_case()

        self.evaluate_difficult = False

        self.tf_pos_lod = [[0, 2, 6]]
        # label score true_pos false_pos
        self.tf_pos = [[1, 0.7, 1, 0], [1, 0.3, 0, 1], [1, 0.2, 1, 0],
                       [2, 0.8, 0, 1], [2, 0.1, 1, 0], [3, 0.2, 0, 1]]


class TestDetectionMAPOp11Point(TestDetectionMAPOp):
    def init_test_case(self):
        super(TestDetectionMAPOp11Point, self).init_test_case()

        self.ap_type = "11point"


if __name__ == '__main__':
    unittest.main()
