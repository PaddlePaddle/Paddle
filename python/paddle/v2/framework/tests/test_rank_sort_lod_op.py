import unittest
import numpy as np
import paddle.v2.framework.core as core


class TestRankSortLoDOpTest(unittest.TestCase):
    def setUp(self):
        scope = core.Scope()
        tensor = scope.var("X").get_tensor()
        tensor.set_dims(10, 1)
        tensor.set([i for i in range(10)], core.CPUPlace())
        tensor.set_lod([[0, 3, 4, 10]])

        scope.var("Out")
        op = core.Operator("rank_sort_lod", X="X", Out="Out", lod_level=0)
        ctx = core.DeviceContext.create(core.CPUPlace())
        op.run(scope, ctx)

        out = np.array(scope.var("Out").get_tensor())
        target = np.array([
            [4, 10, 2],
            [0, 3, 0],
            [3, 4, 1],
        ])

        print 'out', out
        self.assertTrue(np.isclose(out, target).all())


if __name__ == '__main__':
    unittest.main()
