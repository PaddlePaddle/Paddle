import unittest

import numpy as np
import paddle.v2.framework.core as core
from paddle.v2.framework.op import Operator


class TestBeamSearchDecodeOp(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        self.cpu_place = core.CPUPlace()

    def generate_test_tensor_array(self, name, data_type):
        assert data_type in ['float32', 'int64']

        var = self.scope.var(name)
        tensor_array = var.get_lod_tensor_array()

        t0 = core.LoDTensor()
        t0.set(np.array([1], dtype=data_type), self.cpu_place)
        t0.set_lod([[0, 1]])
        tensor_array.append(t0)

        t1 = core.LoDTensor()
        t1.set(np.array([1], dtype=data_type), self.cpu_place)
        t1.set_lod([[0, 1]])
        tensor_array.append(t1)

        t2 = core.LoDTensor()
        t2.set(np.array([1], dtype=data_type), self.cpu_place)
        t2.set_lod([[0, 1]])
        tensor_array.append(t2)

        return tensor_array

    def test_get_set(self):
        ids = self.generate_test_tensor_array("ids", "int64")
        scores = self.generate_test_tensor_array("scores", "float32")

        sentence_ids = self.scope.var("sentence_ids").get_tensor()
        sentence_scores = self.scope.var("sentence_scores").get_tensor()

        beam_search_decode_op = Operator(
            "beam_search_decode",
            # inputs
            Ids="ids",
            Scores="scores",
            # outputs
            SentenceIds="sentence_ids",
            SentenceScores="sentence_scores")

        ctx = core.DeviceContext.create(self.cpu_place)
        beam_search_decode_op.run(self.scope, ctx)


if __name__ == '__main__':
    unittest.main()
