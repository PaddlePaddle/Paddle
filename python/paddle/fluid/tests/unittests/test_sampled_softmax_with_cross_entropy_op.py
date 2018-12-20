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
import numpy as np
from op_test import OpTest
from test_softmax_op import stable_softmax


class Sampler(object):
    def __init__(self, range):
        self.range_ = range

    def sample(self):
        rasie("No Implementation!")

    def probability(self, value):
        raise ("No Implementation!")


class UniformSampler(Sampler):
    def sample(self):
        return np.random.randint(0, self.range_)

    def probability(self, value):
        return 1.0 / self.range_


class LogUniformSampler(Sampler):
    def __init__(self, range):
        super(LogUniformSampler, self).__init__(range)
        self.log_range_ = np.log(self.range_ + 1)

    def sample(self):
        value = int(np.exp(np.random.uniform(0.0, self.logrange_)) - 1)
        return value % self.range_

    def probability(self, value):
        return np.log((value + 2.0) / (value + 1.0)) / self.log_range_


'''
def take_along_axis(array, index):
    data_type = array.dtype
    out = np.zeros_like(index, dtype=data_type)
    nrologits in pythonlogits in pythonlogits in pythonlogits in pythonlogits in pythonlogits in pythonlogits in pythonw, ncol = array.shape
    n_take = index.shape[1]
    for i in range(nrow):
        for j in range(n_take):
            out[i, j] = array[i, index[i, j]]
    return out
    
def put_along_axis(array, index, value):
    nrow, ncol = array.shape
    n_put = index.shape[1]
    for i in range(nrow):
        for j in n_put:
            array[i, index[i, j]] = value[i, j]
'''


def sampled_softmax_with_cross_entropy(logits, label, sampler_type,
                                       num_samples):
    batch_size, num_classes = logits.shape
    num_true = label.shape[1]
    num_sampled_classes = num_true + num_samples
    custom_negative_label = np.tile(np.arange(num_samples), (batch_size, 1))
    samples = np.concatenate((label, custom_negative_label), axis=1)
    #print("logits in python is")
    #print(logits)
    #print("true label in python is")
    #print(label)
    #print("samples in pythn is")
    #print(samples)
    sampler = LogUniformSampler(num_classes) if sampler_type == "log_uniform" \
                else UniformSampler(num_classes)
    probabilities = np.vectorize(sampler.probability)(samples)
    #print("probabilities in python is")
    #print(probabilities)
    sampled_logits = np.take_along_axis(logits, samples, axis=1)
    #print("sampled logits in python is")
    #print(sampled_logits)
    sampled_logits -= np.log(probabilities)
    #print("sampled subtracted logits in python is")
    #print(sampled_logits)
    sampled_softmax = np.apply_along_axis(
        func1d=stable_softmax, axis=1, arr=sampled_logits)
    #print("sampled_softmax in python is")
    #print(sampled_softmax)
    shifted_true_labels = np.tile(np.arange(num_true), (batch_size, 1))
    log_sampled_softmax = np.log(sampled_softmax)
    loss = -np.sum(np.take_along_axis(
        log_sampled_softmax, shifted_true_labels, axis=1),
                   axis=1,
                   keepdims=True) / num_true
    #print("loss in python is")
    #print(loss)
    return (loss, samples, sampled_softmax)


class TestSampledSoftmaxWithCrossEntropy(OpTest):
    def generate_data(self, batch_size, num_classes, num_true, num_samples,
                      sampler_type):
        logits = np.random.randn(batch_size, num_classes).astype(np.float64)
        dodged_range = list(range(num_samples, num_classes))
        label = np.stack([
            np.random.choice(
                dodged_range, num_true, replace=False)
            for _ in range(batch_size)
        ]).astype(np.int64)

        self.attrs = {
            'sampler': sampler_type,
            'num_samples': num_samples,
            'custom_negative_classes': list(range(num_samples))
        }
        self.inputs = {'Logits': logits, 'Label': label}

    def set_data(self):
        self.generate_data(10, 2000, 5, 15, 'uniform')

    def compute(self):
        out = sampled_softmax_with_cross_entropy(
            self.inputs['Logits'], self.inputs['Label'], self.attrs['sampler'],
            self.attrs['num_samples'])
        self.outputs = {
            'Loss': out[0],
            'Samples': out[1],
            'SampledSoftmax': out[2]
        }

    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        self.set_data()
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["Logits", "SampledSoftmax"], "Loss", max_relative_error=0.02)


class TestSampledSoftmaxWithCrossEntropyCase1(
        TestSampledSoftmaxWithCrossEntropy):
    def set_data(self):
        self.generate_data(10, 2000, 5, 25, 'log_uniform')


class TestSampledSoftmaxWithCrossEntropyCase2(
        TestSampledSoftmaxWithCrossEntropy):
    def set_data(self):
        self.generate_data(10, 2000, 5, 60, 'log_uniform')


class TestSampledSoftmaxWithCrossEntropyCase3(
        TestSampledSoftmaxWithCrossEntropy):
    def set_data(self):
        self.generate_data(10, 5000, 5, 128, 'uniform')


class TestSampledSoftmaxWithCrossEntropyCase4(
        TestSampledSoftmaxWithCrossEntropy):
    def set_data(self):
        self.generate_data(30, 5000, 5, 128, 'log_uniform')


if __name__ == '__main__':
    unittest.main()
