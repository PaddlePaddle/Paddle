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

import numpy as np
from math import log
from math import exp
from op_test import OpTest
from scipy.special import logit
from scipy.special import expit
import unittest


class TestTeacherStudentSigmoidLossOp(OpTest):
    """
        Test teacher_student_sigmoid_loss with discrete one-hot labels.
    """

    def setUp(self):
        self.op_type = "teacher_student_sigmoid_loss"
        batch_size = 100
        num_classes = 1
        self.inputs = {
            'X': logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float64")),
            'Label': np.random.uniform(0, 2, (batch_size, num_classes))
            .astype("float64")
        }
        outs = []
        for index, label in enumerate(self.inputs["Label"]):
            x = self.inputs["X"][index]
            if label < -1.0:
                outs.append(max(x, 0.0) + log(1.0 + exp(-abs(x))))
            elif label < 0.0:
                outs.append(max(x, 0.0) - x + log(1.0 + exp(-abs(x))))
            elif label < 1.0:
                outs.append(max(x, 0.0) + log(1.0 + exp(-abs(x))) + \
                            max(x, 0.0) - x * label + log(1.0 + exp(-abs(x))))
            else:
                outs.append(max(x, 0.0) - x + log(1.0 + exp(-abs(x))) + \
                            max(x, 0.0) - x * (label - 1.0) + log(1.0 + exp(-abs(x))))
        self.outputs = {'Y': np.array(outs)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Y", numeric_grad_delta=0.005)
