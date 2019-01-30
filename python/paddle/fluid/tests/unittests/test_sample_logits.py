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

    tmp_samples = set()
    num_tries = 0
    j = 0
    while j < num_true:
        for i in range(batch_size):
            samples[i, j] = label[i, j]
            probabilities[i, j] = sampler.probability(label[i, j])
        j += 1
    while j < num_sampled_classes:
        v = sampler.sample()
        num_tries += 1
        if v not in tmp_samples:
            tmp_samples.add(v)
            for i in range(batch_size):
                samples[i, j] = v
                probabilities[i, j] = sampler.probability(v)
            j += 1
    for k in range(num_sampled_classes):
        for i in range(batch_size):
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


def sample_logits(logits,
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

    #print(samples)
    #print(probabilities)
    #print(sampled_logits)
    if remove_accidental_hits:
        compute_remove_accidental_hits(sampled_logits, samples, num_true)
    sampled_logits -= np.log(probabilities)
    sampled_label = np.tile(np.arange(num_true), (batch_size, 1))
    return (sampled_logits, samples, sampled_label, probabilities)


class TestSampleLogitsOp(OpTest):
    '''
    Test SampleLogitsOp, but with random results precomputed
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
        out = sample_logits(self.inputs["Logits"], self.inputs["Label"],
                            self.attrs["num_samples"], self.attrs["seed"],
                            self.attrs["remove_accidental_hits"],
                            self.attrs["use_custom_samples"],
                            self.inputs["CustomSamples"],
                            self.inputs["CustomProbabilities"])

        self.outputs = {
            'SampledLogits': out[0],
            'Samples': out[1],
            'SampledLabel': out[2],
            'Probabilities': out[3]
        }

    def setUp(self):
        self.op_type = 'sample_logits'
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
            ["Logits"], ["SampledLogits", "Samples"], max_relative_error=0.02)


class TestSampleLogitsOp2(TestSampleLogitsOp):
    def setUp(self):
        self.op_type = 'sample_logits'
        batch_size = 5
        num_classes = 20
        num_true = 5
        num_samples = 10
        seed = 10
        remove_accidental_hits = False
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampleLogitsOp3(TestSampleLogitsOp):
    def setUp(self):
        self.op_type = 'sample_logits'
        batch_size = 5
        num_classes = 100
        num_true = 5
        num_samples = 25
        seed = 10
        remove_accidental_hits = True
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampleLogitsOp4(TestSampleLogitsOp):
    def setUp(self):
        self.op_type = 'sample_logits'
        batch_size = 5
        num_classes = 100
        num_true = 5
        num_samples = 25
        seed = 10
        remove_accidental_hits = False
        self.set_data(batch_size, num_classes, num_true, num_samples, seed,
                      remove_accidental_hits)
        self.compute()


class TestSampleLogitsOpV2(OpTest):
    '''
    Test SampleLogitsOp, but with random results precomputed
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

    def set_data(self, num_classes, num_samples, seed, remove_accidental_hits):
        label = np.array([[6, 12, 15, 5, 1], [0, 9, 4, 1, 10],
                          [0, 2, 10, 16, 13], [14, 4, 7, 2, 1],
                          [3, 18, 11, 8, 14]])
        batch_size, num_true = label.shape
        use_custom_samples = False

        num_sampled_classes = num_samples + num_true
        logits = np.random.randn(batch_size, num_classes)

        remove_accidental_hits = remove_accidental_hits
        self.generate_data(logits, label, num_samples, seed,
                           remove_accidental_hits, use_custom_samples)

        # python and c++ use different random generator
        # use fetched samples from c++ for python code
        self.fetched_samples = np.array(
            [[6, 12, 15, 5, 1, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [0, 9, 4, 1, 10, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [0, 2, 10, 16, 13, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [14, 4, 7, 2, 1, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4],
             [3, 18, 11, 8, 14, 5, 15, 1, 0, 8, 3, 14, 2, 13, 4]])
        fectched_num_tries = 21

        probabilities = np.zeros(
            (batch_size, num_sampled_classes), dtype=np.float64)

        sampler = LogUniformSampler(num_classes, seed)
        for j in range(num_sampled_classes):
            for i in range(batch_size):
                probabilities[i, j] = sampler.probability(self.fetched_samples[
                    i, j])
                probabilities[i, j] = adjust_prob(
                    probabilities[i, j], num_samples, fectched_num_tries)
        self.probabilities = probabilities

    def compute(self):
        out = sample_logits(self.inputs["Logits"], self.inputs["Label"],
                            self.attrs["num_samples"], self.attrs["seed"],
                            self.attrs["remove_accidental_hits"], True,
                            self.fetched_samples, self.probabilities)
        self.outputs = {
            'SampledLogits': out[0],
            'Samples': out[1],
            'SampledLabel': out[2],
            'Probabilities': out[3]
        }

    def setUp(self):
        self.op_type = 'sample_logits'
        num_samples = 10
        num_classes = 20
        seed = 10
        remove_accidental_hits = True

        self.set_data(num_classes, num_samples, seed, remove_accidental_hits)
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        self.check_grad(
            ["Logits"], ["SampledLogits", "Samples"], max_relative_error=0.02)


class TestSampleLogitsOpV3(OpTest):
    '''
    Test SampleLogitsOp, but with random results precomputed
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

    def set_data(self, num_classes, num_samples, seed, remove_accidental_hits):
        self.fetched_samples = np.array([[
            52,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            2,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            2,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            17,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            96,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            2,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            17,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            96,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            37,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ], [
            2,
            3,
            12,
            74,
            28,
            1,
            79,
            2,
            42,
            8,
            13,
            0,
            18,
            88,
            49,
            14,
            46,
            39,
            57,
            26,
            75,
            9,
            50,
            16,
            66,
            6,
            23,
            5,
            11,
            17,
            54,
            35,
            20,
            53,
            10,
            47,
            80,
            38,
            7,
            4,
            31,
            15,
            19,
            58,
            22,
            34,
            41,
            73,
            62,
            95,
            25,
            70,
            37,
            30,
            65,
            27,
            51,
            43,
            32,
            99,
            21,
            56,
            29,
            40,
            69,
            55,
            98,
            77,
            67,
            33,
            89,
            63,
            81,
            59,
            48,
            91,
            68,
            72,
            61,
            52,
            86,
        ]])
        fectched_num_tries = 323

        label = self.fetched_samples[:, 0:1]
        batch_size, num_true = label.shape
        use_custom_samples = False

        #import pdb; pdb.set_trace()
        num_sampled_classes = num_samples + num_true
        logits = np.random.randn(batch_size, num_classes)

        remove_accidental_hits = remove_accidental_hits
        self.generate_data(logits, label, num_samples, seed,
                           remove_accidental_hits, use_custom_samples)

        # python and c++ use different random generator
        # use fetched samples from c++ for python code
        probabilities = np.zeros(
            (batch_size, num_sampled_classes), dtype=np.float64)

        sampler = LogUniformSampler(num_classes, seed)
        for j in range(num_sampled_classes):
            for i in range(batch_size):
                probabilities[i, j] = sampler.probability(self.fetched_samples[
                    i, j])
                probabilities[i, j] = adjust_prob(
                    probabilities[i, j], num_samples, fectched_num_tries)
        self.probabilities = probabilities

    def compute(self):
        out = sample_logits(self.inputs["Logits"], self.inputs["Label"],
                            self.attrs["num_samples"], self.attrs["seed"],
                            self.attrs["remove_accidental_hits"], True,
                            self.fetched_samples, self.probabilities)
        self.outputs = {
            'SampledLogits': out[0],
            'Samples': out[1],
            'SampledLabel': out[2],
            'Probabilities': out[3]
        }

    def setUp(self):
        self.op_type = 'sample_logits'
        num_samples = 80
        num_classes = 100
        seed = 123
        remove_accidental_hits = True

        self.set_data(num_classes, num_samples, seed, remove_accidental_hits)
        self.compute()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass
        self.check_grad(
            ["Logits"], ["SampledLogits", "Samples"], max_relative_error=0.02)


if __name__ == '__main__':
    unittest.main()
