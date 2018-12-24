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
    def __init__(self, range, seed):
        self.range_ = range
        self.seed_ = seed
        np.random.seed(self.seed_)

    def sample(self):
        rasie("No Implementation!")

    def probability(self, value):
        raise ("No Implementation!")


class LogUniformSampler(Sampler):
    def __init__(self, range, seed):
        super(LogUniformSampler, self).__init__(range, seed)
        self.log_range_ = np.log(self.range_ + 1)

    def sample(self):
        value = int(np.exp(np.random.uniform(0.0, self.log_range_)) - 1)
        return value % self.range_

    def probability(self, value):
        return np.log((value + 2.0) / (value + 1.0)) / self.log_range_


def adjust_prob(prob, num_samples, num_tries):
    if num_samples == num_tries:
        return prob * num_samples
    else:
        return -np.expm1(num_tries * np.log1p(-prob))


def take_along_axis1(array, index):
    out = np.zeros_like(index, dtype=array.dtype)
    n_row, n_col = index.shape
    for i in range(n_row):
        for j in range(n_col):
            out[i, j] = array[i, index[i, j]]
    return out


def sample_prob(sampler, num_samples, label):
    batch_size, num_true = label.shape
    num_sampled_classes = num_samples + num_true

    samples = np.zeros((batch_size, num_sampled_classes), dtype=np.int64)
    probabilities = np.zeros(
        (batch_size, num_sampled_classes), dtype=np.float64)
    num_tries_vec = []

    for i in range(batch_size):
        tmp_samples = set()
        tmp_true_labels = set(label[i])
        num_tries = 0
        j = 0
        while j < num_true:
            samples[i, j] = label[i, j]
            probabilities[i, j] = sampler.probability(label[i, j])
            j += 1
        while j < num_sampled_classes:
            v = sampler.sample()
            num_tries += 1
            if v not in tmp_samples and v not in tmp_true_labels:
                tmp_samples.add(v)
                samples[i, j] = v
                probabilities[i, j] = sampler.probability(v)
                j += 1
        num_tries_vec.append(num_tries)
        for k in range(num_sampled_classes):
            probabilities[i, k] = adjust_prob(probabilities[i, k], num_samples,
                                              num_tries)
    return (samples, probabilities)


def sampled_softmax_with_cross_entropy(logits, label, num_samples, seed,
                                       use_custom_samples, custom_samples,
                                       custom_probabilities):
    batch_size, num_classes = logits.shape
    num_true = label.shape[1]
    num_sampled_classes = num_true + num_samples

    if use_custom_samples:
        samples = custom_samples
        probabilities = custom_probabilities
    else:
        sampler = LogUniformSampler(num_classes, seed)
        samples, probabilities = sample_prob(sampler, num_samples, labels)
    sampled_logits = take_along_axis1(logits, samples)
    sampled_logits -= np.log(probabilities)
    sampled_softmax = np.apply_along_axis(
        func1d=stable_softmax, axis=1, arr=sampled_logits)
    shifted_true_labels = np.tile(np.arange(num_true), (batch_size, 1))
    log_sampled_softmax = np.log(sampled_softmax)
    loss = -np.sum(take_along_axis1(log_sampled_softmax, shifted_true_labels),
                   axis=1,
                   keepdims=True) / num_true
    return (loss, samples, sampled_softmax)


class TestSampledSoftmaxWithCrossEntropy(OpTest):
    '''
    Test SampledSoftmaxWithCrossEntropyOp, but with random results precomputed
    in C++ and copied to python and just test the non-random part.
    '''

    def generate_data(self, logits, label, num_samples, seed,
                      use_custom_samples, custom_samples, custom_probabilities):
        self.attrs = {
            'num_samples': num_samples,
            'use_custom_samples': use_custom_samples,
            'seed': seed
        }
        self.inputs = {
            'Logits': logits,
            'Label': label,
            'CustomSamples': custom_samples,
            'CustomProbabilities': custom_probabilities
        }

    def set_data(self, batch_size, num_classes, num_true, num_samples, seed):
        logits = np.random.randn(batch_size, num_classes)
        label = np.stack([
            np.random.choice(
                range(0, num_classes), num_true, replace=False)
            for _ in range(batch_size)
        ])
        sampler = LogUniformSampler(num_classes, seed)
        custom_samples, custom_probabilities = \
            sample_prob(sampler, num_samples, label)
        use_custom_samples = True
        self.generate_data(logits, label, num_samples, seed, use_custom_samples,
                           custom_samples, custom_probabilities)

    def compute(self):
        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["num_samples"], self.attrs["seed"],
            self.attrs["use_custom_samples"], self.inputs["CustomSamples"],
            self.inputs["CustomProbabilities"])

        self.outputs = {
            'Loss': out[0],
            'Samples': out[1],
            'SampledSoftmax': out[2]
        }

    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        batch_size = 5
        num_classes = 20
        num_true = 5
        num_samples = 10
        seed = 10
        self.set_data(batch_size, num_classes, num_true, num_samples, seed)
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        self.check_grad(
            ["Logits", "SampledSoftmax"], "Loss", max_relative_error=0.02)


if __name__ == '__main__':
    unittest.main()
