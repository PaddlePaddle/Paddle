import logging
import paddle.v2.framework.core as core
import unittest
import numpy as np


class TestTensorArray(unittest.TestCase):
    def setUp(self):
        self.ta = core.TensorArray()

        self.batch_size = 10
        self.dim = 2

        # create a LoDTensor
        self.scope = core.Scope()
        var = self.scope.new_var("test_tensor")
        self.place = core.CPUPlace()
        tensor = var.get_tensor()
        tensor.set_dims([self.batch_size, self.dim])
        tensor.alloc_float(self.place)
        tensor_array = np.array(tensor)
        tensor_array[0, 0] = 0
        tensor_array[1, 0] = 1
        tensor_array[2, 0] = 2
        tensor_array[3, 0] = 3
        tensor_array[4, 0] = 4
        tensor_array[5, 0] = 5
        tensor_array[6, 0] = 6
        tensor_array[7, 0] = 7
        tensor_array[8, 0] = 8
        tensor_array[9, 0] = 9

        lod_py = [[0, 2, 5, 10]]
        lod_tensor = core.LoDTensor(lod_py)
        lod_tensor.set(tensor_array, self.place)

        self.py_seq_meta = [[5, 10, 2], [2, 5, 1], [0, 2, 0]]

        self.tensor = lod_tensor

    def test_unstack(self):
        self.ta.unstack(self.tensor)
        self.assertEqual(self.tensor.get_dims()[0], self.ta.size())

    def test_read(self):
        self.ta.unstack(self.tensor)
        for i in range(self.batch_size):
            tensor = self.ta.read(i)

    def test_write(self):
        self.ta.unstack(self.tensor)

        # create a tensor with shape of [1, self.dim]
        var = self.scope.new_var("hell")
        tensor = var.get_tensor()
        tensor.set_dims([1, self.dim])
        tensor.alloc_float(self.place)
        tensor_array = np.array(tensor)
        for i in range(self.dim):
            tensor_array[0, i] = i
        tensor.set(tensor_array, self.place)

        self.ta.write(2, tensor)

        ta_tensor = self.ta.read(2)
        ta_tensor_array = np.array(ta_tensor)
        self.assertEqual(ta_tensor.get_dims(), [1, self.dim])
        self.assertTrue((tensor_array == ta_tensor_array).all())

    def test_write_shared(self):
        self.ta.unstack(self.tensor)

        # create a tensor with shape of [1, self.dim]
        var = self.scope.new_var("hell")
        tensor = var.get_tensor()
        tensor.set_dims([1, self.dim])
        tensor.alloc_float(self.place)
        tensor_array = np.array(tensor)
        for i in range(self.dim):
            tensor_array[0, i] = i
        tensor.set(tensor_array, self.place)

        self.ta.write_shared(2, tensor)

        ta_tensor = self.ta.read(2)
        ta_tensor_array = np.array(ta_tensor)
        self.assertEqual(ta_tensor.get_dims(), [1, self.dim])
        self.assertTrue((tensor_array == ta_tensor_array).all())

    def test_unpack(self):
        meta = self.ta.unpack(self.tensor, 0, True)
        self.assertEqual(self.ta.size(), 5)
        self.assertEqual(meta, self.py_seq_meta)

    def test_pack(self):
        meta = self.ta.unpack(self.tensor, 0, True)
        print "meta", meta
        tensor = self.ta.pack(0, meta, self.tensor.lod())
        print np.array(self.tensor)
        print np.array(tensor)
        self.assertTrue((np.array(self.tensor) == np.array(tensor)).all())
        self.assertTrue(tensor.lod(), self.tensor.lod())


if __name__ == '__main__':
    unittest.main()
