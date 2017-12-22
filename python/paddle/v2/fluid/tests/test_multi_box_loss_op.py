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

        # dim = {2, 4, 1, 1}
        loc = [ [ [[0.1]], [[0.1]], [[0.1]], [[0.1]] ],
                [ [[0.1]], [[0.1]], [[0.1]], [[0.1]] ] ]
        loc0 = np.array(loc).astype('float32')
        loc1 = np.array(loc).astype('float32')

        # dim = {2, 2, 1, 1}
        conf = [[[[0.1]], [[0.9]]],
                [[[0.2]], [[0.8]]]]
        conf0 = np.array(conf).astype('float32')
        conf1 = np.array(conf).astype('float32')

        self.loc = [('loc0', loc0), ('loc1', loc1)]
        self.conf = [('conf0', conf0), ('conf1', conf1)]

        self.prior_box = [0.1, 0.1, 0.5, 0.5,
                          0.1, 0.1, 0.2, 0.2,
                          0.2, 0.2, 0.6, 0.6,
                          0.1, 0.1, 0.2, 0.2,
                          0.3, 0.3, 0.7, 0.7,
                          0.1, 0.1, 0.2, 0.2,
                          0.4, 0.4, 0.8, 0.8,
                          0.1, 0.1, 0.2, 0.2]
        self.prior_box = np.array(self.prior_box).astype('float32')


        self.label_lod = [[0, 2, 4]]
        self.label = [[1, 0.1, 0.1, 0.3, 0.3, 0],
                      [1, 0.6, 0.6, 0.8, 0.8, 1],
                      [2, 0.3, 0.3, 0.6, 0.5, 0],
                      [1, 0.7, 0.1, 0.9, 0.3, 0]]

        self.label = np.array(self.label).astype('float32')

        self.inter_counter = [4, 4, 8]
        self.inter_counter = np.array(
            self.inter_counter).flatten().astype('int64')


        self.all_match_indices = [[0, -1, -1,  1],
                                  [-1,  0, 0, -1]]
        self.all_match_indices = np.array(
            self.all_match_indices).astype('int64')

        self.all_neg_indices = [[2,   1, -1, -1],
                                [0,   3, -1, -1]]
        self.all_neg_indices = np.array(
            self.all_neg_indices).astype('int64')

        self.loc_gt_data = [[-2.50],  [-2.50], [-3.47],  [-3.47],
                            [2.50],   [2.50],  [-3.47],  [-3.47],
                            [1.25],  [0.00],   [-1.44], [-3.47],
                            [-1.25], [-2.50], [-1.44], [-3.47]]
        self.loc_gt_data = np.array(
            self.loc_gt_data).astype('float32')
        
        self.conf_gt_data = [[1], [1], [0], [0], [2], [2], [0], [0]]
        self.conf_gt_data = np.array(
            self.conf_gt_data).astype('int64')

        self.loc_diff = [[0.1], [0.1], [0.1], [0.1],
                         [0.1], [0.1], [0.1], [0.1],
                         [0.1], [0.1], [0.1], [0.1],
                         [0.0], [0.0], [0.0], [0.0]]
        self.loc_diff = np.array(self.loc_diff).astype('float32')

        self.conf_prob = [[0.231,  0.514,  0.255],
                          [0.333,  0.333,  0.333],
                          [0.275,  0.500,  0.225],
                          [0.384,  0.190,  0.424],
                          [0.333,  0.333,  0.333],
                          [0.333,  0.333,  0.333],
                          [0.333,  0.333,  0.333],
                          [0.333,  0.333,  0.333]]
        self.conf_prob = np.array(self.conf_prob).astype('float32')

        self.loss = np.array([9.55]).flatten().astype('float32')

    def setUp(self):
        self.op_type = "multi_box_loss"
        self.set_data()

    def test_check_output(self):
        self.check_output(atol=0.01)

    def test_check_grad(self):
        return
        self.check_grad(['loc0'], 'Loss')

if __name__ == '__main__':
    unittest.main()
