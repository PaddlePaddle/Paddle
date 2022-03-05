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

from __future__ import print_function

import unittest
import paddle
import numpy as np
import sys
sys.path.append("..")
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class XPUTestLabelSmoothOp(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'label_smooth'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestLabelSmoothOp
        classes = []
        batch_sizes = [1, 5, 1024]
        label_dims = [1, 7, 12]
        for bs in batch_sizes:
            for label_dim in label_dims:
                class_name = 'XPUTestLabelSmooth_' + \
                       str(bs) + "_" + str(label_dim)
                attr_dict = {'batch_size': bs, 'label_dim': label_dim}
                classes.append([class_name, attr_dict])
        classes.append(['XPUTestLabelSmooth_3d', {'is_3d': True}])
        return base_class, classes

    class TestLabelSmoothOp(XPUOpTest):
        def setUp(self):
            self.op_type = "label_smooth"
            self.epsilon = 0.1
            self.use_xpu = True
            if not hasattr(self, 'batch_size'):
                self.batch_size = 10
                self.label_dim = 12
            self.label = np.zeros(
                (self.batch_size, self.label_dim)).astype("float32")
            nonzero_index = np.random.randint(
                self.label_dim, size=(self.batch_size))
            self.label[np.arange(self.batch_size), nonzero_index] = 1
            smoothed_label = (1 - self.epsilon
                              ) * self.label + self.epsilon / self.label_dim
            self.inputs = {'X': self.label}
            self.attrs = {'epsilon': self.epsilon}
            self.outputs = {'Out': smoothed_label}
            if hasattr(self, 'is_3d') and self.is_3d:
                self.inputs['X'] = self.inputs['X'].reshape(
                    [2, -1, self.inputs['X'].shape[-1]])
                self.outputs['Out'] = self.outputs['Out'].reshape(self.inputs[
                    'X'].shape)

        def test_check_output(self):
            if not paddle.is_compiled_with_xpu():
                return
            self.check_output_with_place(paddle.XPUPlace(0), atol=1e-6)

        def test_check_grad(self):
            return


support_types = get_xpu_op_support_types('label_smooth')
for stype in support_types:
    create_test_class(globals(), XPUTestLabelSmoothOp, stype)

if __name__ == '__main__':
    unittest.main()
