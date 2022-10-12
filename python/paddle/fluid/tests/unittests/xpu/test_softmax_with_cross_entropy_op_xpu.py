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

import sys

sys.path.append("..")

from test_softmax_op import stable_softmax
from op_test_xpu import XPUOpTest
import paddle.fluid.core as core
import paddle

import unittest
import numpy as np
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


class XPUTestSoftmaxWithCrossEntropyOp(XPUOpTestWrapper):

    def __init__(self):
        self.op_name = 'softmax_with_cross_entropy'
        self.use_dynamic_create_class = True

    def dynamic_create_class(self):
        base_class = self.TestSoftmaxWithCrossEntropyOp
        classes = []
        shapes = [[41, 37], [3, 5, 7, 11], [3, 5, 7, 1], [1023, 38512],
                  [1, 511]]
        for soft_label in [True, False]:
            for numeric_stable_mode in [True, False]:
                for shape in shapes:
                    for logits_type in [0, 1, 2]:
                        for axis in range(len(shape)):
                            if (not numeric_stable_mode):
                                axis = -1
                            class_name = 'XPUTestSoftmaxWithCrossEntropy_' + \
                                   str(soft_label) + "_" + \
                                   str(numeric_stable_mode) + "_" + \
                                   str(shape) + "_" + \
                                   str(logits_type) + "_" + \
                                   str(axis)
                            attr_dict = {'soft_label': soft_label, \
                                         'numeric_stable_mode': numeric_stable_mode, \
                                         'shape': shape, \
                                         'logits_type': logits_type,
                                         'axis': axis}
                            classes.append([class_name, attr_dict])
        return base_class, classes

    class TestSoftmaxWithCrossEntropyOp(XPUOpTest):
        """
        Test softmax with cross entropy operator with discreate one-hot labels.
        """

        def setUp(self):
            self.op_type = "softmax_with_cross_entropy"
            self.use_xpu = True
            self.dtype = np.float32
            self.ignore_index = -1

            if not hasattr(self, 'shape'):
                self.shape = [43, 6]
                self.numeric_stable_mode = True
                self.logits_type = 0
                self.soft_label = True
                self.axis = -1
            logits = getattr(
                self, "logits",
                np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype))
            if self.logits_type == 1:
                self.logits = np.full(self.shape, -500.0).astype(self.dtype)
            elif self.logits_type == 2 and len(self.shape) == 4:
                self.logits = np.full(self.shape, 1000.0).astype(self.dtype)
                self.logits[:, :, 0, :] = -1000.0
            softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

            if self.soft_label:
                labels = np.random.uniform(0.1, 1.0,
                                           self.shape).astype(self.dtype)
                labels /= np.sum(labels, axis=self.axis, keepdims=True)
            else:
                axis_dim = self.shape[self.axis]
                self.shape[self.axis] = 1
                labels = np.random.randint(0,
                                           axis_dim,
                                           self.shape,
                                           dtype="int64")

            loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                                 self.ignore_index)

            self.inputs = {"Logits": logits, "Label": labels}
            self.outputs = {
                "Softmax": softmax.astype(self.dtype),
                "Loss": loss.astype(self.dtype)
            }
            self.attrs = {
                "numeric_stable_mode": self.numeric_stable_mode,
                "soft_label": self.soft_label,
            }
            if self.ignore_index >= 0:
                self.attrs['ignore_index'] = self.ignore_index
            if self.axis != -1:
                self.attrs['axis'] = self.axis

        def test_check_output(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_output_with_place(place, atol=1e-2)

        def test_check_grad(self):
            if paddle.is_compiled_with_xpu():
                paddle.enable_static()
                place = paddle.XPUPlace(0)
                self.check_grad_with_place(place, ["Logits"],
                                           "Loss",
                                           max_relative_error=0.2)


support_types = get_xpu_op_support_types('softmax_with_cross_entropy')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftmaxWithCrossEntropyOp, stype)

if __name__ == "__main__":
    unittest.main()
