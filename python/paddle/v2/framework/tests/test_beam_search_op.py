import logging
from paddle.v2.framework.op import Operator, DynamicRecurrentOp
import paddle.v2.framework.core as core
import unittest
import numpy as np


def create_tensor(scope, name, np_data):
    tensor = scope.var(name).get_tensor()
    tensor.set(np_data, core.CPUPlace())
    return tensor


class BeamSearchOpTester(unittest.TestCase):
    def setUp(self):
        self.scope = core.Scope()
        self.ctx = core.DeviceContext.create(core.CPUPlace())
        self._create_ids()
        self._create_scores()
        self.scope.var('selected_ids')
        self.scope.var('selected_scores')

    def test_run(self):
        op = Operator(
            'beam_search',
            pre_ids="pre_ids",
            ids='ids',
            scores='scores',
            selected_ids='selected_ids',
            selected_scores='selected_scores',
            level=0,
            beam_size=2,
            end_id=0, )
        op.run(self.scope, self.ctx)
        selected_ids = self.scope.find_var("selected_ids").get_tensor()
        print 'selected_ids', np.array(selected_ids)
        print 'lod', selected_ids.lod()

    def _create_pre_ids(self):
        np_data = np.array([[1, 2, 3, 4]])
        tensor = create_tensor(self.scope, "pre_ids", np_data)

    def _create_ids(self):
        self.lod = [[0, 1, 4], [0, 1, 2, 3, 4]]
        np_data = np.array(
            [[4, 2, 5], [2, 1, 3], [3, 5, 2], [8, 2, 1]], dtype='int32')
        tensor = create_tensor(self.scope, "ids", np_data)
        tensor.set_lod(self.lod)

    def _create_scores(self):
        np_data = np.array(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.9, 0.5, 0.1],
                [0.7, 0.5, 0.1],
            ],
            dtype='float32')
        tensor = create_tensor(self.scope, "scores", np_data)
        tensor.set_lod(self.lod)


if __name__ == '__main__':
    unittest.main()
