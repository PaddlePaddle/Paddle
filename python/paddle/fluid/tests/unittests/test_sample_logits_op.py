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

import unittest
import collections
import numpy as np
from op_test import OpTest


class TestSampleLogitsOp(OpTest):
    def setUp(self):
        self.op_type = "sample_logits"
        self.dtype = np.float64
        self.use_mkldnn = False
        bs = 2
        K = 20
        NT = 10
        S = 5

        Samples = np.random.random([bs, NT + S]).astype('int64')
        Probabilities = np.random.random([bs, NT + S]).astype('float64')
        LogitsDim = np.array([bs, K], dtype=np.int64)
        LabelsDim = np.array([bs, NT], dtype=np.int64)
        SampledLogits = np.random.random([bs, NT + S]).astype('float64')
        SampledLabels = np.random.random([bs, NT]).astype('int64')

        self.bs = bs
        self.K = K
        self.NT = NT
        self.S = S
        Labels = np.array(list(range(self.NT)) * self.bs).astype('int64')
        self.Labels = Labels.reshape(self.bs, -1)
        self.Logits = np.random.random([self.bs, self.K]).astype('float64')

        self.inputs = {"Logits": self.Logits, "Labels": self.Labels}
        self.fetch_list = [
            'Samples', 'Probabilities', 'SampledLogits', 'SampledLabels'
        ]
        self.outputs = collections.OrderedDict(
            (('Samples', Samples), ('Probabilities', Probabilities),
             ('LogitsDim', LogitsDim), ('LabelsDim', LabelsDim),
             ('SampledLogits', SampledLogits), ('SampledLabels',
                                                SampledLabels)))

        self.attrs = {'num_samples': self.S}

    def test_check_output(self):
        places = self._get_places()
        for p in places:
            (Samples, Probabilities, SampledLogits,
             SampledLabels) = [np.array(o) for o in self.calc_output(p)]

            assert Samples.dtype == np.int64, \
                "Samples dtype is {}, not int64".format(Samples.dtype)
            assert Probabilities.dtype == np.float64, \
                "Probabilities dtype is {}, not float64".format(
                    Probabilities.dtype)
            assert SampledLogits.dtype == np.float64, \
                "SampledLogits dtype is {}, not float64".format(
                    SampledLogits.dtype)
            assert SampledLabels.dtype == np.int64, \
                "SampledLabels dtype is {}, not int64".format(
                    SampledLabels.dtype)

            assert Samples.shape == (self.bs, self.NT + self.S)
            assert Probabilities.shape == (self.bs, self.NT + self.S)
            assert SampledLogits.shape == (self.bs, self.NT + self.S)
            assert SampledLabels.shape == (self.bs, self.NT)

            assert (SampledLabels == self.Labels).all()
            sampled_logits = self.Logits[:, Samples[0][:self.NT]]
            sampled_logits -= np.log(Probabilities[:, :self.NT])
            np.testing.assert_almost_equal(sampled_logits,
                                           SampledLogits[:, :self.NT])

    def test_check_grad(self):
        self._check_grad_helper()
        for p in self._get_places():
            grads = self._get_gradient(['Logits'], p, ['SampledLogits'], [])
            np.testing.assert_almost_equal(grads[0].sum(), np.array([1.]))


class TestSampleLogitsOpNoUniq(TestSampleLogitsOp):
    def setUp(self):
        super(TestSampleLogitsOpNoUniq, self).setUp()
        self.attrs = {'num_samples': self.S, 'uniq': False}


class TestSampleLogitsOpWithAccidentalHits(TestSampleLogitsOp):
    def setUp(self):
        super(TestSampleLogitsOpWithAccidentalHits, self).setUp()
        self.attrs = {'num_samples': self.S, 'remove_accidental_hits': False}


if __name__ == "__main__":
    unittest.main()
