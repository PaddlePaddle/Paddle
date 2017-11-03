import unittest
import paddle.v2.framework.core as core
import numpy
import paddle.v2.framework.layers as layers
from paddle.v2.framework.framework import Program
from paddle.v2.framework.executor import Executor


class TestCPULoDTensorArrayOps(unittest.TestCase):
    def place(self):
        return core.CPUPlace()

    def test_lod_tensor_to_array_level_0(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 3, 9, 10]])
        expect = map(lambda x: numpy.array(x).astype('int32'),
                     [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]])
        self.main(tensor=tensor, expect_array=expect, expect_lod=[] * 6)

    def main(self, tensor, expect_array, expect_lod):
        place = self.place()
        program = Program()
        x = layers.data(name='x', shape=[10], program=program)
        table = layers.lod_rank_table(x, level=0, program=program)
        array = layers.lod_tensor_to_array(x, table, program=program)
        array.persistable = True

        result = layers.array_to_lod_tensor(array, table, program=program)
        result.persistable = True
        exe = Executor(place)
        scope = core.Scope()
        exe.run(program, feed={'x': tensor}, scope=scope)
        var = scope.find_var(array.name)
        array = var.get_lod_tensor_array()
        self.check_array_same(array, expect_array, expect_lod)
        self.check_tensor_same(scope.find_var(x.name).get_tensor(), tensor)

    def check_array_same(self, array, expect_tensor, expect_lod):
        self.assertEqual(len(expect_tensor), len(array))
        for i, exp in enumerate(zip(expect_tensor, expect_lod)):
            exp_tensor, exp_lod = exp
            self.assertEqual(exp_tensor, numpy.array(array[i]))
            self.assertEqual(exp_lod, array[i].get_lod())

    def check_tensor_same(self, actual, expect):
        self.assertEqual(numpy.array(actual), numpy.array(expect))
        self.assertEqual(actual.get_lod(), expect.get_lod())


if __name__ == '__main__':
    unittest.main()
