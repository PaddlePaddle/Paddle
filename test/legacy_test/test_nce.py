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


def nce(
    input, weight, bias, sample_weight, labels, num_classes, num_sample_class
):
    samples = []
    sample_labels = []
    batch_size = input.shape[0]
    num_true_class = labels.shape[1]
    for i in range(batch_size):
        w = 1 if sample_weight is None else sample_weight[i]
        for label in labels[i]:
            samples.append((i, label, True, w))
            sample_labels.append(label)
        for num in range(num_sample_class):
            samples.append((i, num, False, w))
            sample_labels.append(num)
    # forward bias
    sample_out = np.zeros(len(samples)).astype(np.float32)
    if bias is not None:
        for i in range(len(samples)):
            sample_out[i] = bias[samples[i][1]]
    # forward weight
    for i in range(len(samples)):
        sample_out[i] += np.dot(input[samples[i][0]], weight[samples[i][1]])

    # forward activation
    sample_out = 1.0 / (1.0 + np.exp(-sample_out))
    # forward cost
    out = np.zeros(batch_size).astype(np.float32)
    b = 1.0 / num_classes * num_sample_class
    for i in range(len(samples)):
        o = sample_out[i]
        cost = -np.log(o / (o + b)) if samples[i][2] else -np.log(b / (o + b))
        out[samples[i][0]] += cost * samples[i][3]
    return (
        out[:, np.newaxis],
        np.array(sample_out).reshape(
            batch_size, num_sample_class + num_true_class
        ),
        np.array(sample_labels).reshape(
            batch_size, num_sample_class + num_true_class
        ),
    )


class TestNCE(OpTest):
    def generate_data(
        self,
        dim,
        batch_size,
        num_classes,
        num_true_class,
        num_neg_samples,
        is_sparse,
    ):
        input = np.random.randn(batch_size, dim).astype(np.float32)
        weight = np.random.randn(num_classes, dim).astype(np.float32)
        bias = np.random.randn(num_classes).astype(np.float32)
        sample_weight = np.random.randn(batch_size).astype(np.float32)
        labels = np.random.randint(
            0, num_classes, (batch_size, num_true_class)
        ).astype("int64")
        self.attrs = {
            'num_total_classes': num_classes,
            'num_neg_samples': num_neg_samples,
            'custom_neg_classes': list(range(num_neg_samples)),
            'seed': 0,
            'sampler': 0,
            'is_sparse': is_sparse,
            'is_test': self.is_test,
        }
        self.inputs = {
            'Input': input,
            'Label': labels,
            'Weight': weight,
            'Bias': bias,
            'SampleWeight': sample_weight,
        }

    def set_is_test(self):
        self.is_test = False

    def set_data(self):
        self.generate_data(5, 25, 100, 1, 2, False)

    def compute(self):
        out = nce(
            self.inputs['Input'],
            self.inputs['Weight'],
            self.inputs['Bias'],
            self.inputs['SampleWeight'],
            self.inputs['Label'],
            self.attrs['num_total_classes'],
            self.attrs['num_neg_samples'],
        )
        if self.is_test:
            self.outputs = {'Cost': out[0]}
        else:
            self.outputs = {
                'Cost': out[0],
                'SampleLogits': out[1],
                'SampleLabels': out[2],
            }

    def setUp(self):
        self.op_type = 'nce'
        self.set_is_test()
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output(check_dygraph=False)

    def test_check_grad(self):
        self.check_grad(
            ["Input", "Weight", "Bias"],
            "Cost",
            max_relative_error=0.02,
            check_dygraph=False,
        )


class TestNCECase1Tensor(TestNCE):
    def set_data(self):
        self.generate_data(10, 20, 100, 2, 5, False)


class TestNCETensorIsTest(TestNCE):
    # if is_test = True, there's no need to calculate grad
    def set_is_test(self):
        self.is_test = True

    def test_check_grad(self):
        pass


if __name__ == '__main__':
    unittest.main()
