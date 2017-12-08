from paddle.v2.fluid.layers import lod_rank_table, data
from paddle.v2.fluid.executor import Executor
import paddle.v2.fluid.core as core
import numpy
import unittest


class TestLoDRankTable(unittest.TestCase):
    def test_lod_rank_table(self):
        x = data(name='x', shape=[100])
        cpu = core.CPUPlace()
        rank_table = lod_rank_table(x=x, level=1)
        rank_table.persistable = True
        exe = Executor(cpu)
        scope = core.Scope()

        tensor = core.LoDTensor()
        tensor.set(numpy.random.random(size=(17, 100)), cpu)
        tensor.set_lod([[0, 1, 3], [0, 5, 6, 7], [0, 3, 4, 9, 10, 13, 16, 17]])
        exe.run(scope=scope, feed={'x': tensor})
        var = scope.find_var(rank_table.name)
        table = var.get_lod_rank_table()
        self.assertEqual([(0, 5), (1, 1), (2, 1)], table.items())


if __name__ == '__main__':
    unittest.main()
