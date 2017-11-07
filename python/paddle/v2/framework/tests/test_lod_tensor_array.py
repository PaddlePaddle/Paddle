import unittest
import paddle.v2.framework.core as core
import numpy


class TestLoDTensorArray(unittest.TestCase):
    def test_get_set(self):
        scope = core.Scope()
        arr = scope.var('tmp_lod_tensor_array')
        tensor_array = arr.get_lod_tensor_array()
        self.assertEqual(0, len(tensor_array))
        cpu = core.CPUPlace()
        for i in xrange(10):
            t = core.LoDTensor()
            t.set(numpy.array([i], dtype='float32'), cpu)
            t.set_lod([[0, 1]])
            tensor_array.append(t)

        self.assertEqual(10, len(tensor_array))

        for i in xrange(10):
            t = tensor_array[i]
            self.assertEqual(numpy.array(t), numpy.array([i], dtype='float32'))
            self.assertEqual([[0, 1]], t.lod())

            t = core.LoDTensor()
            t.set(numpy.array([i + 10], dtype='float32'), cpu)
            t.set_lod([[0, 2]])
            tensor_array[i] = t
            t = tensor_array[i]
            self.assertEqual(
                numpy.array(t), numpy.array(
                    [i + 10], dtype='float32'))
            self.assertEqual([[0, 2]], t.lod())


if __name__ == '__main__':
    unittest.main()
