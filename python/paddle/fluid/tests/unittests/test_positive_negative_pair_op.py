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
import itertools
import numpy as np
import six
from op_test import OpTest


def py_pnpair_op(score, label, query, column=-1, weight=None):
    # group by query id
    predictions = {}
    batch_size = label.shape[0]
    if weight is None:
        weight = np.ones(shape=(batch_size, 1)).astype('float32')
    for s, l, q, w in zip(score, label, query, weight):
        s, l, q, w = s[column], l[0], q[0], w[0]
        if q not in predictions:
            predictions[q] = []
        predictions[q].append((s, l, w))

    # accumulate statistics
    pos, neg, neu = 0, 0, 0
    for _, ranks in six.iteritems(predictions):
        for e1, e2 in itertools.combinations(ranks, 2):
            s1, s2, l1, l2, w1, w2 = e1[0], e2[0], e1[1], e2[1], e1[2], e2[2]
            w = (w1 + w2) * 0.5
            if l1 == l2:
                continue
            if s1 == s2:
                neu += w
            elif (s1 - s2) * (l1 - l2) > 0:
                pos += w
            else:
                neg += w

    return np.array(pos).astype('float32'), np.array(neg).astype(
        'float32'), np.array(neu).astype('float32')


class TestPositiveNegativePairOp(OpTest):

    def setUp(self):
        self.op_type = 'positive_negative_pair'
        batch_size = 20
        max_query_id = 5
        score = np.random.normal(size=(batch_size, 1)).astype('float32')
        label = np.random.normal(size=(batch_size, 1)).astype('float32')
        query = np.array(
            [np.random.randint(max_query_id) for i in range(batch_size)])
        query = np.reshape(query, newshape=(batch_size, 1)).astype('int64')

        pos, neg, neu = py_pnpair_op(score, label, query)
        self.inputs = {'Score': score, 'Label': label, 'QueryID': query}
        self.attrs = {'column': -1}
        self.outputs = {
            'PositivePair': pos,
            'NegativePair': neg,
            'NeutralPair': neu
        }

    def test_check_output(self):
        self.check_output()


class TestPositiveNegativePairOpAccumulateWeight(OpTest):

    def setUp(self):
        self.op_type = 'positive_negative_pair'
        batch_size = 20
        max_query_id = 5
        max_random_num = 2 << 15
        score_dim = 2
        score = np.random.normal(size=(batch_size, 2)).astype('float32')
        label = np.random.normal(size=(batch_size, 1)).astype('float32')
        weight = np.random.normal(size=(batch_size, 1)).astype('float32')
        query = np.array(
            [np.random.randint(max_query_id) for i in range(batch_size)])
        query = np.reshape(query, newshape=(batch_size, 1)).astype('int64')
        acc_pos = np.reshape(np.random.randint(max_random_num),
                             newshape=(1)).astype('float32')
        acc_neg = np.reshape(np.random.randint(max_random_num),
                             newshape=(1)).astype('float32')
        acc_neu = np.reshape(np.random.randint(max_random_num),
                             newshape=(1)).astype('float32')
        column = np.random.randint(score_dim)

        pos, neg, neu = py_pnpair_op(score,
                                     label,
                                     query,
                                     column=column,
                                     weight=weight)
        self.inputs = {
            'Score': score,
            'Label': label,
            'QueryID': query,
            'AccumulatePositivePair': acc_pos,
            'AccumulateNegativePair': acc_neg,
            'AccumulateNeutralPair': acc_neu,
            'Weight': weight
        }
        self.attrs = {'column': column}
        self.outputs = {
            'PositivePair': pos + acc_pos,
            'NegativePair': neg + acc_neg,
            'NeutralPair': neu + acc_neu
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
