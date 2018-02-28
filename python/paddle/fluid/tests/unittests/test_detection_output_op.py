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


class TestUnpoolOp(OpTest):
    def setUp(self):
        self.op_type = "detection_output"
        self.init_test_case()

        #loc.shape ((1, 4, 4, 1, 1))
        #conf.shape ((1, 4, 2, 1, 1))

        loc = np.array([[[[[0.1]], [[0.1]], [[0.1]], [[0.1]]],
                         [[[0.1]], [[0.1]], [[0.1]], [[0.1]]],
                         [[[0.1]], [[0.1]], [[0.1]], [[0.1]]],
                         [[[0.1]], [[0.1]], [[0.1]], [[0.1]]]]])
        conf = np.array([[[[[0.1]], [[0.9]]], [[[0.2]], [[0.8]]],
                          [[[0.3]], [[0.7]]], [[[0.4]], [[0.6]]]]])
        priorbox = np.array([
            0.1, 0.1, 0.5, 0.5, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.6, 0.6, 0.1,
            0.1, 0.2, 0.2, 0.3, 0.3, 0.7, 0.7, 0.1, 0.1, 0.2, 0.2, 0.4, 0.4,
            0.8, 0.8, 0.1, 0.1, 0.2, 0.2
        ])

        output = np.array([
            0, 1, 0.68997443, 0.099959746, 0.099959746, 0.50804031, 0.50804031
        ])
        self.inputs = {
            'Loc': loc.astype('float32'),
            'Conf': conf.astype('float32'),
            'PriorBox': priorbox.astype('float32')
        }
        self.attrs = {
            'num_classes': self.num_classes,
            'top_k': self.top_k,
            'nms_top_k': self.nms_top_k,
            'background_label_id': self.background_label_id,
            'nms_threshold': self.nms_threshold,
            'confidence_threshold': self.confidence_threshold,
        }
        self.outputs = {'Out': output.astype('float32')}

    def test_check_output(self):
        self.check_output()

    def init_test_case(self):
        self.num_classes = 2
        self.top_k = 10
        self.nms_top_k = 20
        self.background_label_id = 0
        self.nms_threshold = 0.01
        self.confidence_threshold = 0.01


if __name__ == '__main__':
    unittest.main()
