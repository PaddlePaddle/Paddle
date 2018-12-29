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

    for i in range(batch_size):
        tmp_samples = set()
        num_tries = 0
        j = 0
        while j < num_true:
            samples[i, j] = label[i, j]
            probabilities[i, j] = sampler.probability(label[i, j])
            j += 1
        while j < num_sampled_classes:
            v = sampler.sample()
            num_tries += 1
            if v not in tmp_samples:
                tmp_samples.add(v)
                samples[i, j] = v
                probabilities[i, j] = sampler.probability(v)
                j += 1
        for k in range(num_sampled_classes):
            probabilities[i, k] = adjust_prob(probabilities[i, k], num_samples,
                                              num_tries)
    return (samples, probabilities)


def compute_remove_accidental_hits(sampled_logits, samples, num_true):
    batch_size, num_sampled_classes = samples.shape
    for i in range(batch_size):
        true_labels = set(samples[i, np.arange(num_true)])
        for j in range(num_true, num_sampled_classes):
            if samples[i, j] in true_labels:
                sampled_logits[i, j] -= 1e20


def sampled_softmax_with_cross_entropy(logits,
                                       label,
                                       num_samples,
                                       seed,
                                       remove_accidental_hits,
                                       use_custom_samples,
                                       custom_samples=None,
                                       custom_probabilities=None):
    batch_size, num_classes = logits.shape
    num_true = label.shape[1]
    num_sampled_classes = num_true + num_samples

    if use_custom_samples:
        samples = custom_samples
        probabilities = custom_probabilities
    else:
        sampler = LogUniformSampler(num_classes, seed)
        samples, probabilities = sample_prob(sampler, num_samples, label)
    sampled_logits = take_along_axis1(logits, samples)

    if remove_accidental_hits:
        compute_remove_accidental_hits(sampled_logits, samples, num_true)
    sampled_logits -= np.log(probabilities)
    sampled_softmax = np.apply_along_axis(
        func1d=stable_softmax, axis=1, arr=sampled_logits)
    shifted_true_labels = np.tile(np.arange(num_true), (batch_size, 1))
    loss = -np.sum(
        np.log(take_along_axis1(sampled_softmax, shifted_true_labels)),
        axis=1,
        keepdims=True) / num_true
    return (loss, samples, sampled_softmax)


class TestSampledSoftmaxWithCrossEntropyOp(OpTest):
    '''
    Test SampledSoftmaxWithCrossEntropyOp, but with random results precomputed
    in python and just test the non-random part.
    '''

    def generate_data(self, logits, label, num_samples, seed,
                      remove_accidental_hits, use_custom_samples,
                      custom_samples, custom_probabilities):
        self.attrs = {
            'num_samples': num_samples,
            'use_custom_samples': use_custom_samples,
            'remove_accidental_hits': remove_accidental_hits,
            'seed': seed
        }
        self.inputs = {
            'Logits': logits,
            'Label': label,
            'CustomSamples': custom_samples,
            'CustomProbabilities': custom_probabilities
        }

    def set_data(self, batch_size, num_classes, num_true, num_samples, seed,
                 remove_accidental_hits):
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
        remove_accidental_hits = remove_accidental_hits
        self.generate_data(logits, label, num_samples, seed,
                           remove_accidental_hits, use_custom_samples,
                           custom_samples, custom_probabilities)

    def compute(self):
        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["num_samples"], self.attrs["seed"],
            self.attrs["remove_accidental_hits"],
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
        remove_accidental_hits = True
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        self.check_grad(
            ["Logits", "SampledSoftmax"], "Loss", max_relative_error=0.02)


class TestSampledSoftmaxWithCrossEntropyOp2(
        TestSampledSoftmaxWithCrossEntropyOp):
    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        batch_size = 5
        num_classes = 20
        num_true = 5
        num_samples = 10
        seed = 10
        remove_accidental_hits = False
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampledSoftmaxWithCrossEntropyOp3(
        TestSampledSoftmaxWithCrossEntropyOp):
    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        batch_size = 5
        num_classes = 100
        num_true = 5
        num_samples = 25
        seed = 10
        remove_accidental_hits = True
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampledSoftmaxWithCrossEntropyOp4(
        TestSampledSoftmaxWithCrossEntropyOp):
    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        batch_size = 5
        num_classes = 100
        num_true = 5
        num_samples = 25
        seed = 10
        remove_accidental_hits = False
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampledSoftmaxWithCrossEntropyOpV2(OpTest):
    '''
    Test SampledSoftmaxWithCrossEntropyOp, but with random results precomputed
    in C++ and copied to python and just test the non-random part.
    '''

    def generate_data(self, logits, label, num_samples, seed,
                      remove_accidental_hits, use_custom_samples):
        self.attrs = {
            'num_samples': num_samples,
            'use_custom_samples': use_custom_samples,
            'remove_accidental_hits': remove_accidental_hits,
            'seed': seed
        }
        self.inputs = {'Logits': logits, 'Label': label}

    def set_data(self, num_samples, seed, remove_accidental_hits):
        logits = np.array([[
            -1.26920285, -0.25028848, 0.69650066, 0.64749644, -0.94378878,
            -0.73634908, -0.40555151, 0.71724483, -1.20622355, 0.17403311,
            -0.07016852, 1.74592598, 0.27176874, -0.49562088, -1.33643965,
            0.88879927, 0.35672408, 0.0643708, -0.56231286, -1.86481024
        ], [
            -0.47797525, -0.54639648, 0.18964547, -0.24959924, 0.42368044,
            1.01914792, -0.87489165, -0.12964218, 1.78845536, -1.84398742,
            -0.37228166, 1.04043935, -0.24349669, 0.0472615, 0.0472767,
            1.48522851, 1.89477055, -1.35981026, 0.83291423, 0.18916936
        ], [
            0.04443455, -0.81648076, -0.24090526, 0.2626387, -1.08181684,
            -1.69116843, -0.12944931, 0.08403161, 0.02438071, -0.34117507,
            0.42280389, -2.07347357, 1.08010437, -1.33319261, 0.60098998,
            0.64670401, 0.22640284, 0.67728021, -1.45828894, -0.37162258
        ], [
            -0.31861864, 1.17645835, 1.20674212, -0.44510221, -1.46600632,
            -0.84797627, 0.13177668, -0.46205261, -1.20948384, -0.6351612,
            0.5124433, 0.87278519, 1.65143622, 0.58072902, 0.89104646,
            1.16666346, -1.01649964, -0.64278475, 0.58792172, -1.52389564
        ], [
            -1.2178384, -0.85469283, 0.22707807, -1.05062937, -0.00464435,
            0.72018271, -1.06269798, -0.8271736, -0.87192148, -0.0298758,
            -0.30460663, -0.72121477, -0.43721224, -1.40163852, 0.63628158,
            1.24281116, -0.52251437, 1.67032187, -1.01046289, 0.74044828
        ]])
        label = np.array([[6, 12, 15, 5, 1], [0, 9, 4, 1, 10],
                          [0, 2, 10, 16, 13], [14, 4, 7, 2, 1],
                          [3, 18, 11, 8, 14]])
        use_custom_samples = False
        remove_accidental_hits = remove_accidental_hits
        self.generate_data(logits, label, num_samples, seed,
                           remove_accidental_hits, use_custom_samples)

    def compute(self):
        fetched_samples = np.array(
            [[6, 12, 15, 5, 1, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [0, 9, 4, 1, 10, 7, 10, 0, 1, 2, 8, 5, 3, 15, 17],
             [0, 2, 10, 16, 13, 2, 3, 1, 0, 4, 5, 6, 17, 14, 15],
             [14, 4, 7, 2, 1, 0, 8, 6, 2, 15, 17, 11, 7, 16, 4],
             [3, 18, 11, 8, 14, 12, 5, 3, 2, 1, 13, 18, 4, 9, 6]])
        fetched_probabilities = np.array([[
            0.610098, 0.403988, 0.344519, 0.664166, 0.950281, 0.664166,
            0.344519, 0.950281, 0.995596, 0.5227, 0.797797, 0.362339, 0.875623,
            0.382059, 0.726599
        ], [
            0.996599, 0.503281, 0.742972, 0.956902, 0.471605, 0.580215,
            0.471605, 0.996599, 0.956902, 0.887375, 0.539218, 0.68117, 0.812617,
            0.357571, 0.325784
        ], [
            0.997971, 0.907654, 0.501376, 0.365466, 0.423124, 0.907654,
            0.839079, 0.967617, 0.997971, 0.772834, 0.712638, 0.659184,
            0.349518, 0.402038, 0.382902
        ], [
            0.274861, 0.603983, 0.446684, 0.774375, 0.882795, 0.979249, 0.41039,
            0.4897, 0.774375, 0.260443, 0.235686, 0.329437, 0.446684, 0.247451,
            0.603983
        ], [
            0.567123, 0.170477, 0.254028, 0.321191, 0.209973, 0.237435,
            0.435349, 0.567123, 0.664404, 0.792401, 0.222866, 0.170477,
            0.493021, 0.295217, 0.389426
        ]])

        out = sampled_softmax_with_cross_entropy(
            self.inputs["Logits"], self.inputs["Label"],
            self.attrs["num_samples"], self.attrs["seed"],
            self.attrs["remove_accidental_hits"], True, fetched_samples,
            fetched_probabilities)
        self.outputs = {
            'Loss': out[0],
            'Samples': out[1],
            'SampledSoftmax': out[2]
        }

    def setUp(self):
        self.op_type = 'sampled_softmax_with_cross_entropy'
        num_samples = 10
        seed = 10
        remove_accidental_hits = True
        self.set_data(num_samples, seed, remove_accidental_hits)
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        self.check_grad(
            ["Logits", "SampledSoftmax"], "Loss", max_relative_error=0.02)


if __name__ == '__main__':
    unittest.main()
