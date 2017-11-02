import unittest
import numpy as np
from op_test import OpTest


def Levenshtein(hyp, ref):
    """ Compute the Levenshtein distance between two strings.

    :param hyp: hypothesis string in index
    :type hyp: list
    :param ref: reference string in index
    :type ref: list
    """
    m = len(hyp)
    n = len(ref)
    if m == 0:
        return n
    if n == 0:
        return m

    dist = np.zeros((m + 1, n + 1))
    for i in range(0, m + 1):
        dist[i][0] = i
    for j in range(0, n + 1):
        dist[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if hyp[i - 1] == ref[j - 1] else 1
            deletion = dist[i - 1][j] + 1
            insertion = dist[i][j - 1] + 1
            substitution = dist[i - 1][j - 1] + cost
            dist[i][j] = min(deletion, insertion, substitution)
    return dist[m][n]


class TestCTCEditDistanceOp(OpTest):
    def setUp(self):
        self.op_type = "ctc_edit_distance"
        normalized = True
        x1 = np.array([0, 12, 3, 5]).astype("int64")
        x2 = np.array([0, 12, 4, 7, 8]).astype("int64")

        distance = Levenshtein(hyp=x1, ref=x2)
        if normalized is True:
            distance = distance / len(x2)
        self.attrs = {'normalized': normalized}
        self.inputs = {'X1': x1, 'X2': x2}
        self.outputs = {'Out': distance}

    def test_check_output(self):
        self.check_output()


if __name__ == '__main__':
    unittest.main()
