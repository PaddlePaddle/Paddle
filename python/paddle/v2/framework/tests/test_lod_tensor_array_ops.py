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

    def test_lod_tensor_to_array_level_0_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(10).reshape(10, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 3, 9, 9, 10]])
        expect = map(lambda x: numpy.array(x).astype('int32'),
                     [[3, 0, 9], [4, 1], [5, 2], [6], [7], [8]])
        self.main(tensor=tensor, expect_array=expect, expect_lod=[] * 6)

    def test_lod_tensor_to_array_level_1(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(20).reshape(20, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5], [0, 3, 9, 11, 17, 20]])

        expect = [
            numpy.array(
                [9, 10, 0, 1, 2], dtype='int32'), numpy.array(
                    [11, 12, 13, 14, 15, 16, 3, 4, 5, 6, 7, 8], dtype='int32'),
            numpy.array(
                [17, 18, 19], dtype='int32')
        ]

        lod = [[[0, 2, 5]], [[0, 6, 12]], [[0, 3]]]
        self.main(tensor=tensor, expect_array=expect, expect_lod=lod)

    def test_lod_tensor_to_array_level_1_empty_seq(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(31).reshape(31, 1).astype('int32'), self.place())

        tensor.set_lod([[0, 3, 5, 9, 11],
                        [0, 3, 7, 11, 11, 12, 17, 19, 21, 23, 30, 31]])

        expect = [
            numpy.array(
                item, dtype='int32')
            for item in [[
                12, 13, 14, 15, 16, 0, 1, 2, 23, 24, 25, 26, 27, 28, 29
            ], [17, 18, 3, 4, 5, 6, 11, 30], [19, 20, 7, 8, 9, 10], [21, 22]]
        ]

        lod = [[[0, 5, 8, 8, 15]], [[0, 2, 6, 7, 8]], [[0, 2, 6]], [[0, 2]]]
        self.main(tensor=tensor, expect_array=expect, expect_lod=lod)

    def test_lod_tensor_to_array_level_2(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5, 6], [0, 2, 5, 6, 10, 12, 13],
                        [0, 3, 7, 11, 17, 21, 22, 23, 27, 31, 39, 45, 46, 50]])

        expect = [
            numpy.array(
                item, dtype='int32')
            for item in [[21, 0, 1, 2, 3, 4, 5, 6, 46, 47, 48, 49], range(
                22, 39) + range(7, 21), range(39, 46)]
        ]
        lod = [[[0, 1, 3, 4], [0, 1, 4, 8, 12]],
               [[0, 4, 7], [0, 1, 5, 9, 16, 21, 27, 31]], [[0, 2], [0, 6, 7]]]
        self.main(tensor=tensor, expect_array=expect, expect_lod=lod)

    def test_lod_tensor_to_array_level_2_skip_level(self):
        tensor = core.LoDTensor()
        tensor.set(
            numpy.arange(50).reshape(50, 1).astype('int32'), self.place())
        tensor.set_lod([[0, 2, 5, 6], [0, 2, 5, 6, 10, 12, 13],
                        [0, 3, 7, 11, 17, 21, 22, 23, 27, 31, 39, 45, 46, 50]])
        expect = [
            numpy.array(
                item, dtype='int32')
            for item in [[21, 0, 1, 2, 3, 4, 5, 6, 46, 47, 48, 49], range(
                22, 39) + range(7, 21), range(39, 46)]
        ]
        lod = [[0, 1, 4, 8, 12], [0, 1, 5, 9, 17, 21, 27, 31], [0, 6, 7]]
        self.main(tensor=tensor, expect_array=expect, expect_lod=lod, level=1)

    def main(self, tensor, expect_array, expect_lod, level=0):
        place = self.place()
        program = Program()
        x = layers.data(name='x', shape=[10], main_program=program)
        x.persistable = True
        table = layers.lod_rank_table(x, level=level, main_program=program)
        array = layers.lod_tensor_to_array(x, table, main_program=program)
        array.persistable = True

        result = layers.array_to_lod_tensor(array, table, main_program=program)
        result.persistable = True
        exe = Executor(place)
        scope = core.Scope()
        exe.run(program, feed={'x': tensor}, scope=scope)
        var = scope.find_var(array.name)
        array = var.get_lod_tensor_array()
        self.check_array_same(array, expect_array, expect_lod)
        self.check_tensor_same(scope.find_var(result.name).get_tensor(), tensor)

    def check_array_same(self, array, expect_tensor, expect_lod):
        self.assertEqual(len(expect_tensor), len(array))
        for i, exp in enumerate(zip(expect_tensor, expect_lod)):
            exp_tensor, exp_lod = exp
            self.assertEqual(exp_tensor, numpy.array(array[i]))
            self.assertEqual(exp_lod, array[i].get_lod())

    def check_tensor_same(self, actual, expect):
        self.assertTrue(
            numpy.allclose(numpy.array(actual), numpy.array(expect)))
        self.assertEqual(actual.lod(), expect.lod())


if __name__ == '__main__':
    unittest.main()
