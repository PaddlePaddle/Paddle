import unittest
import numpy as np
import sys
from op_test import OpTest


class TestMultiBoxLossOp(OpTest):
    def set_data(self):
        self.init_test_case()

        self.inputs = {
            'Loc': self.loc,
            'Conf': self.conf,
            'PriorBox': self.prior_box,
            'Label': (self.label, self.label_lod)
        }

        self.attrs = {
            'class_num': self.classes_num,
            'overlap_threshold': self.overlap_threshold,
            'neg_pos_ratio': self.neg_pos_ratio,
            'neg_overlap': self.neg_overlap,
            'background_label_id': self.background_id
        }

        self.outputs = {
            'Loss': self.loss,
            'InterCounter': self.inter_counter,
            'AllMatchIndices': self.all_match_indices,
            'AllNegIndices': self.all_neg_indices,
            'LocGTData': self.loc_gt_data,
            'ConfGTData': self.conf_gt_data,
            'LocDiff': self.loc_diff,
            'ConfProb': self.conf_prob
        }

    def init_test_case(self):
        self.input_num = 2
        self.classes_num = 3
        self.overlap_threshold = 0.3
        self.neg_pos_ratio = 3.0
        self.neg_overlap = 0.5
        self.background_id = 0

        loc0 = [-0.768, -1.032, 0.046, 1.613, -0.205, 2.643, 2.771, 0.207]
        loc1 = [-1.246, 0.096, -0.194, 0.554, -1.722, -2.082, -2.450, 1.673]

        conf0 = [-0.289, -2.602, 0.334, 0.718, -1.706, -2.971]
        conf1 = [-3.235, -2.102, 1.241, -3.959, -1.846, 0.310]

        # dim = {2, 2, 2, 2}
        loc0 = np.array(loc0).reshape((2, 4, 1, 1)).astype('float32')
        loc1 = np.array(loc1).reshape((2, 4, 1, 1)).astype('float32')

        # dim = {2, 3, 2, 2}
        conf0 = np.array(conf0).reshape((2, 3, 1, 1)).astype('float32')
        conf1 = np.array(conf1).reshape((2, 3, 1, 1)).astype('float32')

        self.loc = [('loc0', loc0), ('loc1', loc1)]
        self.conf = [('conf0', conf0), ('conf1', conf1)]

        self.prior_box = [
            0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.6, 0.6, 0.1,
            0.1, 0.2, 0.2
        ]
        self.prior_box = np.array(self.prior_box).astype('float32')

        self.label_lod = [[0, 2, 4]]
        self.label = [[1, 0.1, 0.1, 0.3, 0.3, 0], [1, 0.6, 0.6, 0.8, 0.8, 1],
                      [2, 0.3, 0.3, 0.6, 0.5, 0], [1, 0.7, 0.1, 0.9, 0.3, 0]]

        self.label = np.array(self.label).astype('float32')

        self.inter_counter = [2, 2, 4]
        self.inter_counter = np.array(self.inter_counter).flatten().astype(
            'int64')

        self.all_match_indices = [[0, -1], [-1, 0]]
        self.all_match_indices = np.array(self.all_match_indices).astype(
            'int64')

        self.all_neg_indices = [[1, -1], [0, -1]]
        self.all_neg_indices = np.array(self.all_neg_indices).astype('int64')

        self.loc_gt_data = [[-2.5], [-2.5], [-3.466], [-3.466], [1.25], [0.],
                            [-1.44], [-3.466]]
        self.loc_gt_data = np.array(self.loc_gt_data).astype('float32')

        self.conf_gt_data = [[1], [0], [2], [0]]
        self.conf_gt_data = np.array(self.conf_gt_data).astype('int64')

        self.loc_diff = [[-0.768], [-1.032], [0.046], [1.613], [-1.722],
                         [-2.082], [-2.45], [1.673]]
        self.loc_diff = np.array(self.loc_diff).astype('float32')

        self.conf_prob = [[0.33744144, 0.03339453, 0.62916404],
                          [0.01087106, 0.03375417, 0.95537478],
                          [0.01238802, 0.10248636,
                           0.88512564], [0.89801782, 0.07953442, 0.02244774]]
        self.conf_prob = np.array(self.conf_prob).astype('float32')

        self.loss = np.array([13.57]).flatten().astype('float32')

    def setUp(self):
        self.op_type = "multi_box_loss"
        self.set_data()

    def test_check_output(self):
        self.check_output(atol=0.01)

    def test_check_grad(self):
        self.check_grad(['loc0', 'loc1', 'conf0', 'conf1'], 'Loss')


if __name__ == '__main__':
    unittest.main()
