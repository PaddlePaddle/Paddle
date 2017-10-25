import unittest
import itertools
import numpy as np
from op_test import OpTest


def py_pnpair_op(score, label, query):
    # group by query id
    predictions = {}
    for s, l, q in zip(score, label, query):
        if type(s) is list:
            s = s[-1]
        q = q[0]
        if q not in predictions:
            predictions[q] = []
        predictions[q].append((s, l))

    # accumulate statistics
    pos, neg, neu = 0, 0, 0
    for _, ranks in predictions.items():
        for e1, e2 in itertools.combinations(ranks, 2):
            s1, s2, l1, l2 = e1[0][0], e2[0][0], e1[1][0], e2[1][0]
            if l1 == l2:
                continue
            if s1 == s2:
                neu += 1
            elif (s1 - s2) * (l1 - l2) > 0:
                pos += 1
            else:
                neg += 1

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
        query = np.reshape(query, newshape=(batch_size, 1)).astype('int32')

        pos, neg, neu = py_pnpair_op(score, label, query)
        self.inputs = {}
        self.inputs = {'Score': score, 'Label': label, 'QueryId': query}
        self.outputs = {
            'PositivePair': pos,
            'NegativePair': neg,
            'NeutralPair': neu
        }

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
