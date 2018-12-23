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
    def __init__(self, range, seed):
        super(LogUniformSampler, self).__init__(range, seed)
        self.log_range_ = np.log(self.range_ + 1)

    def sample(self):
        value = int(np.exp(np.random.uniform(0.0, self.logrange_)) - 1)
        return value % self.range_

    def probability(self, value):
        return np.log((value + 2.0) / (value + 1.0)) / self.log_range_


def adjust_prob(unique, prob, num_samples, num_tries):
    if not unique:
        return prob * num_samples
    else:
        return -np.expm1(num_tries * np.log1p(-prob))


def remove_hits(sampled_logits, hits):
    num_sampled_classes = sampled_logits.shape[1]
    for hit in hits:
        i, j = hit // num_sampled_classes, hit % num_sampled_classes
        sampled_logits[i, j] -= 1e20


def pre_computed_sample_prob(unique, sampler, samples, num_tries_vec):
    probabilities = np.vectorize(sampler.probability)(samples)
    batch_size, num_sampled_classes = samples.shape
    for i in range(batch_size):
        for j in range(num_sampled_classes):
            probabilities[i, j] = adjust_prob(unique, probabilities[i, j],
                                              num_sampled_classes,
                                              num_tries_vec[i])
    return samples, probabilities


def sample_prob(sampler, unique, num_samples, remove_accidental_hits, label,
                num_tries_vec, avoid_indices):
    batch_size, num_true = label.shape
    avoid_set = set(avoid_indices)
    num_sampled_classes = num_samples + num_true

    samples = np.zeros((batch_size, num_samples), dtype=np.int64)
    probabilities = np.zeros((batch_size, num_samples), dtype=np.float64)
    hits = []
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

        if unique:
            while j < num_sampled_classes:
                v = sampler.sample()
                num_tries += 1
                if v not in avoid_set and v not in tmp_samples:
                    tmp_samples.add(v)
                    samples[i, j] = v
                    probabilities[i, j] = sampler.probability(v)
                    if remove_accidental_hits and v in tmp_true_labels:
                        hits.append(i * num_sampled_classes + j)
                    j += 1
        else:
            while j < num_sampled_classes:
                v = sampler.sample()
                if v not in avoid_set:
                    samples[i, j] = v
                    probabilities[i, j] = sampler.probability(v)
                    if remove_accidental_hits and v in tmp_true_labels:
                        hits.append(i * num_sampled_classes + j)
                    j += 1
            num_tries = num_samples
        num_tries_vec.append(num_tries)

        for k in range(num_sampled_classes):
            probabilities[i, j] = adjust_prob(unique, probabilities[i, j],
                                              num_samples, num_tries)
    return (samples, probabilities, np.array(
        hits, dtype=np.int64), np.array(
            num_tries_vec, dtype=np.int64))


def sampled_softmax_with_cross_entropy(logits, label, sampler_type, num_samples,
                                       unique, remove_accidental_hits,
                                       avoid_indices, seed, samples, hits,
                                       num_tries_vec):
    batch_size, num_classes = logits.shape
    num_true = label.shape[1]
    num_sampled_classes = num_true + num_samples

    sampler = LogUniformSampler(num_classes, seed) \
                if sampler_type == "log_uniform" \
                else UniformSampler(num_classes, seed)
    samples, probabilities = pre_computed_sample_prob(unique, sampler, samples,
                                                      num_tries_vec)

    sampled_logits = np.take_along_axis(logits, samples, axis=1)
    if remove_accidental_hits:
        remove_hits(sampled_logits, hits)

    sampled_logits -= np.log(probabilities)
    sampled_softmax = np.apply_along_axis(
        func1d=stable_softmax, axis=1, arr=sampled_logits)
    shifted_true_labels = np.tile(np.arange(num_true), (batch_size, 1))

    log_sampled_softmax = np.log(sampled_softmax)
    loss = -np.sum(np.take_along_axis(
        log_sampled_softmax, shifted_true_labels, axis=1),
                   axis=1,
                   keepdims=True) / num_true
    return (loss, samples, sampled_softmax)


class TestSampledSoftmaxWithCrossEntropy(OpTest):
    '''
    Test SampledSoftmaxWithCrossEntropyOp, but with random results precomputed
    in C++ and copied to python and just test the non-random part.
    
    Many details to mention:
    sampling with unique only means that sample are unique, but they may still
    hits true labels.
    sampling with unique=True will compute a num_tries for each example and 
    use it to adjust the probabilities.
    if remove_accidental_hits is True, post processing of the sampled logit will
    be done, i.e. minus those sampled logits whose class hits true labels by 
    1e20.
    '''

    def generate_data(self, logits, label, sampler_type, num_samples, unique,
                      remove_accidental_hits, avoid_indices, seed):
        self.attrs = {
            'sampler_type': sampler_type,
            'num_samples': num_samples,
            'unique': unique,
            'remove_accidental_hits': remove_accidental_hits,
            'avoid_indices': avoid_indices,
            'seed': seed
        }
        self.inputs = {
            'Logits': logits,
            'Label': label,
        }

    def compute(self):
        logits = np.array([[
            -0.16984993, -0.69949395, 0.82078937, 0.29747916, -0.6253932,
            0.14563557, 0.64444638, -0.88937087, -0.5610792, -2.01975667,
            -0.02065311, 1.28813608, 0.08244265, -2.73794951, 0.61963523,
            0.82480575, 0.18876259, 1.338033, -0.16883108, 1.22144607
        ], [
            -0.49696317, -1.05354859, 0.85424585, 0.46430616, 0.79626186,
            -0.15537106, -1.45241345, -0.96195983, 0.62510423, -0.74338598,
            1.86432751, -0.88609675, -1.3166156, 1.35971528, -0.07968084,
            -0.80837612, -0.76798308, 0.4230423, 1.76831161, -2.23210399
        ], [
            2.28366985, 1.09882092, -1.24805549, 0.60760044, 0.05763591,
            -0.30581664, -2.19215257, 0.78079176, 0.66738125, -2.15903332,
            2.41992787, -0.7324027, -0.75368009, 0.88287498, 0.75182741,
            1.99168179, 0.01808637, 1.47329144, -0.97267191, 1.14939686
        ], [
            1.56809263, -0.21568201, -1.58709263, -1.19326454, -0.10154069,
            0.20942226, 0.61830026, 1.25192055, -1.29105451, 1.39896649,
            -0.48345119, 0.03136012, -0.31837211, -1.37509167, -0.55344305,
            -0.96191502, -0.7714157, 0.4860551, 0.58064789, -0.36974309
        ], [
            0.37223223, 0.31122923, 0.52508788, -1.15166496, -0.2661284,
            -0.4154919, 1.15236939, 1.04728794, -0.64127343, 0.22553791,
            -0.65250907, -0.93587344, 0.4997572, -1.51983087, 0.79306681,
            -0.95566413, 0.15944435, 0.83799413, 0.36811002, 0.68399843
        ]])
        label = np.array([[15, 16, 10, 11, 4], [4, 16, 3, 8, 14],
                          [13, 5, 11, 12, 0], [13, 12, 10, 16, 8],
                          [4, 10, 7, 16, 14]])
        sampler_type = 'log_uniform'
        num_samples = 10
        unique = True
        remove_accidental_hits = True
        avoid_indices = []
        seed = 10
        pre_computed_samples = np.array(
            [[15, 16, 10, 11, 4, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [4, 16, 3, 8, 14, 7, 10, 0, 1, 2, 8, 5, 3, 15, 17],
             [13, 5, 11, 12, 0, 2, 3, 1, 0, 4, 5, 6, 17, 14, 15],
             [13, 12, 10, 16, 8, 0, 8, 6, 2, 15, 17, 11, 7, 16, 4],
             [4, 10, 7, 16, 14, 12, 5, 3, 2, 1, 13, 18, 4, 9, 6]])
        pre_computed_hits = np.array([6, 14, 25, 27, 38, 40, 51, 58, 72])
        pre_computed_num_tries = np.array([21, 22, 24, 15, 11])

        self.generate_data(logits, label, sampler_type, num_samples, unique,
                           remove_accidental_hits, avoid_indices, seed)

        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["sampler_type"], self.attrs["num_samples"],
            self.attrs["unique"], self.attrs["remove_accidental_hits"],
            self.attrs["avoid_indices"], self.attrs["seed"],
            pre_computed_samples, pre_computed_hits, pre_computed_num_tries)

        self.outputs = {
            'Loss': out[0],
            'Samples': pre_computed_samples,
            'SampledSoftmax': out[2]
        }

    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            ["Logits", "SampledSoftmax"], "Loss", max_relative_error=0.02)


class TestCase2(TestSampledSoftmaxWithCrossEntropy):
    def compute(self):
        logits = np.array([[
            -0.16984993, -0.69949395, 0.82078937, 0.29747916, -0.6253932,
            0.14563557, 0.64444638, -0.88937087, -0.5610792, -2.01975667,
            -0.02065311, 1.28813608, 0.08244265, -2.73794951, 0.61963523,
            0.82480575, 0.18876259, 1.338033, -0.16883108, 1.22144607
        ], [
            -0.49696317, -1.05354859, 0.85424585, 0.46430616, 0.79626186,
            -0.15537106, -1.45241345, -0.96195983, 0.62510423, -0.74338598,
            1.86432751, -0.88609675, -1.3166156, 1.35971528, -0.07968084,
            -0.80837612, -0.76798308, 0.4230423, 1.76831161, -2.23210399
        ], [
            2.28366985, 1.09882092, -1.24805549, 0.60760044, 0.05763591,
            -0.30581664, -2.19215257, 0.78079176, 0.66738125, -2.15903332,
            2.41992787, -0.7324027, -0.75368009, 0.88287498, 0.75182741,
            1.99168179, 0.01808637, 1.47329144, -0.97267191, 1.14939686
        ], [
            1.56809263, -0.21568201, -1.58709263, -1.19326454, -0.10154069,
            0.20942226, 0.61830026, 1.25192055, -1.29105451, 1.39896649,
            -0.48345119, 0.03136012, -0.31837211, -1.37509167, -0.55344305,
            -0.96191502, -0.7714157, 0.4860551, 0.58064789, -0.36974309
        ], [
            0.37223223, 0.31122923, 0.52508788, -1.15166496, -0.2661284,
            -0.4154919, 1.15236939, 1.04728794, -0.64127343, 0.22553791,
            -0.65250907, -0.93587344, 0.4997572, -1.51983087, 0.79306681,
            -0.95566413, 0.15944435, 0.83799413, 0.36811002, 0.68399843
        ]])
        label = np.array([[15, 16, 10, 11, 4], [4, 16, 3, 8, 14],
                          [13, 5, 11, 12, 0], [13, 12, 10, 16, 8],
                          [4, 10, 7, 16, 14]])
        sampler_type = 'log_uniform'
        num_samples = 10
        unique = True
        remove_accidental_hits = False
        avoid_indices = []
        seed = 10
        pre_computed_samples = np.array(
            [[15, 16, 10, 11, 4, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [4, 16, 3, 8, 14, 7, 10, 0, 1, 2, 8, 5, 3, 15, 17],
             [13, 5, 11, 12, 0, 2, 3, 1, 0, 4, 5, 6, 17, 14, 15],
             [13, 12, 10, 16, 8, 0, 8, 6, 2, 15, 17, 11, 7, 16, 4],
             [4, 10, 7, 16, 14, 12, 5, 3, 2, 1, 13, 18, 4, 9, 6]])
        pre_computed_hits = np.array([6, 14, 25, 27, 38, 40, 51, 58, 72])
        pre_computed_num_tries = np.array([21, 22, 24, 15, 11])

        self.generate_data(logits, label, sampler_type, num_samples, unique,
                           remove_accidental_hits, avoid_indices, seed)

        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["sampler_type"], self.attrs["num_samples"],
            self.attrs["unique"], self.attrs["remove_accidental_hits"],
            self.attrs["avoid_indices"], self.attrs["seed"],
            pre_computed_samples, pre_computed_hits, pre_computed_num_tries)

        self.outputs = {
            'Loss': out[0],
            'Samples': pre_computed_samples,
            'SampledSoftmax': out[2]
        }


class TestCase3(TestSampledSoftmaxWithCrossEntropy):
    def compute(self):
        logits = np.array([[
            -0.16984993, -0.69949395, 0.82078937, 0.29747916, -0.6253932,
            0.14563557, 0.64444638, -0.88937087, -0.5610792, -2.01975667,
            -0.02065311, 1.28813608, 0.08244265, -2.73794951, 0.61963523,
            0.82480575, 0.18876259, 1.338033, -0.16883108, 1.22144607
        ], [
            -0.49696317, -1.05354859, 0.85424585, 0.46430616, 0.79626186,
            -0.15537106, -1.45241345, -0.96195983, 0.62510423, -0.74338598,
            1.86432751, -0.88609675, -1.3166156, 1.35971528, -0.07968084,
            -0.80837612, -0.76798308, 0.4230423, 1.76831161, -2.23210399
        ], [
            2.28366985, 1.09882092, -1.24805549, 0.60760044, 0.05763591,
            -0.30581664, -2.19215257, 0.78079176, 0.66738125, -2.15903332,
            2.41992787, -0.7324027, -0.75368009, 0.88287498, 0.75182741,
            1.99168179, 0.01808637, 1.47329144, -0.97267191, 1.14939686
        ], [
            1.56809263, -0.21568201, -1.58709263, -1.19326454, -0.10154069,
            0.20942226, 0.61830026, 1.25192055, -1.29105451, 1.39896649,
            -0.48345119, 0.03136012, -0.31837211, -1.37509167, -0.55344305,
            -0.96191502, -0.7714157, 0.4860551, 0.58064789, -0.36974309
        ], [
            0.37223223, 0.31122923, 0.52508788, -1.15166496, -0.2661284,
            -0.4154919, 1.15236939, 1.04728794, -0.64127343, 0.22553791,
            -0.65250907, -0.93587344, 0.4997572, -1.51983087, 0.79306681,
            -0.95566413, 0.15944435, 0.83799413, 0.36811002, 0.68399843
        ]])
        label = np.array([[15, 16, 10, 11, 4], [4, 16, 3, 8, 14],
                          [13, 5, 11, 12, 0], [13, 12, 10, 16, 8],
                          [4, 10, 7, 16, 14]])
        sampler_type = 'uniform'
        num_samples = 10
        unique = False
        remove_accidental_hits = False
        avoid_indices = []
        seed = 10
        pre_computed_samples = np.array(
            [[15, 16, 10, 11, 4, 12, 18, 5, 12, 0, 15, 9, 18, 6, 7],
             [4, 16, 3, 8, 14, 2, 18, 7, 2, 18, 12, 17, 9, 4, 17],
             [13, 5, 11, 12, 0, 11, 14, 16, 1, 5, 8, 4, 6, 3, 14],
             [13, 12, 10, 16, 8, 6, 6, 4, 13, 0, 12, 10, 4, 18, 7],
             [4, 10, 7, 16, 14, 3, 14, 19, 8, 10, 5, 6, 3, 6, 5]])
        pre_computed_hits = np.array([])
        pre_computed_num_tries = np.array([10, 10, 10, 10, 10])

        self.generate_data(logits, label, sampler_type, num_samples, unique,
                           remove_accidental_hits, avoid_indices, seed)

        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["sampler_type"], self.attrs["num_samples"],
            self.attrs["unique"], self.attrs["remove_accidental_hits"],
            self.attrs["avoid_indices"], self.attrs["seed"],
            pre_computed_samples, pre_computed_hits, pre_computed_num_tries)

        self.outputs = {
            'Loss': out[0],
            'Samples': pre_computed_samples,
            'SampledSoftmax': out[2]
        }


if __name__ == '__main__':
    unittest.main()
