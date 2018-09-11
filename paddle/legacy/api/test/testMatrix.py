# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from py_paddle import swig_paddle
import numpy as np
import unittest


class TestMatrix(unittest.TestCase):
    def test_createZero_get_set(self):
        m = swig_paddle.Matrix.createZero(32, 24)
        self.assertEqual(m.getWidth(), 24)
        self.assertEqual(m.getHeight(), 32)
        for x in xrange(24):
            for y in xrange(32):
                self.assertEqual(0.0, m.get(x, y))
        with self.assertRaises(swig_paddle.RangeError):
            m.get(51, 47)
        m.set(3, 3, 3.0)
        self.assertEqual(m.get(3, 3), 3.0)

    def test_sparse(self):
        m = swig_paddle.Matrix.createSparse(3, 3, 6, True, False, False)
        self.assertIsNotNone(m)
        self.assertTrue(m.isSparse())
        self.assertEqual(m.getSparseValueType(), swig_paddle.SPARSE_NON_VALUE)
        self.assertEqual(m.getSparseFormat(), swig_paddle.SPARSE_CSR)
        m.sparseCopyFrom([0, 2, 3, 3], [0, 1, 2], [])
        self.assertEqual(m.getSparseRowCols(0), [0, 1])
        self.assertEqual(m.getSparseRowCols(1), [2])
        self.assertEqual(m.getSparseRowCols(2), [])

    def test_sparse_value(self):
        m = swig_paddle.Matrix.createSparse(3, 3, 6, False, False, False)
        self.assertIsNotNone(m)
        m.sparseCopyFrom([0, 2, 3, 3], [0, 1, 2], [7.3, 4.2, 3.2])

        def assertKVArraySame(actual, expect):
            self.assertEqual(len(actual), len(expect))
            for i in xrange(len(actual)):
                a = actual[i]
                e = expect[i]
                self.assertIsInstance(a, tuple)
                self.assertIsInstance(e, tuple)
                self.assertEqual(len(a), 2)
                self.assertEqual(len(e), 2)
                self.assertEqual(a[0], e[0])
                self.assertTrue(abs(a[1] - e[1]) < 1e-5)

        first_row = m.getSparseRowColsVal(0)
        assertKVArraySame(first_row, [(0, 7.3), (1, 4.2)])

    def test_createDenseMat(self):
        m = swig_paddle.Matrix.createDense([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 2, 3)
        self.assertIsNotNone(m)
        self.assertTrue(abs(m.get(1, 1) - 0.5) < 1e-5)

    def test_numpyCpu(self):
        numpy_mat = np.matrix([[1, 2], [3, 4], [5, 6]], dtype="float32")
        m = swig_paddle.Matrix.createCpuDenseFromNumpy(numpy_mat, False)
        self.assertEqual((int(m.getHeight()), int(m.getWidth())),
                         numpy_mat.shape)

        # the numpy matrix and paddle matrix shared the same memory.
        numpy_mat[0, 1] = 342.23

        for h in xrange(m.getHeight()):
            for w in xrange(m.getWidth()):
                self.assertEqual(m.get(h, w), numpy_mat[h, w])

        mat2 = m.toNumpyMatInplace()
        mat2[1, 1] = 32.2
        self.assertTrue(np.array_equal(mat2, numpy_mat))

    def test_numpyGpu(self):
        if swig_paddle.isGpuVersion():
            numpy_mat = np.matrix([[1, 2], [3, 4], [5, 6]], dtype='float32')
            gpu_m = swig_paddle.Matrix.createGpuDenseFromNumpy(numpy_mat)
            assert isinstance(gpu_m, swig_paddle.Matrix)
            self.assertEqual((int(gpu_m.getHeight()), int(gpu_m.getWidth())),
                             numpy_mat.shape)
            self.assertTrue(gpu_m.isGpu())
            numpy_mat = gpu_m.copyToNumpyMat()
            numpy_mat[0, 1] = 3.23
            for a, e in zip(gpu_m.getData(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
                self.assertAlmostEqual(a, e)

            gpu_m.copyFromNumpyMat(numpy_mat)

            for a, e in zip(gpu_m.getData(), [1.0, 3.23, 3.0, 4.0, 5.0, 6.0]):
                self.assertAlmostEqual(a, e)

    def test_numpy(self):
        numpy_mat = np.matrix([[1, 2], [3, 4], [5, 6]], dtype="float32")
        m = swig_paddle.Matrix.createDenseFromNumpy(numpy_mat)
        self.assertEqual((int(m.getHeight()), int(m.getWidth())),
                         numpy_mat.shape)
        self.assertEqual(m.isGpu(), swig_paddle.isUsingGpu())
        for a, e in zip(m.getData(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]):
            self.assertAlmostEqual(a, e)


if __name__ == "__main__":
    swig_paddle.initPaddle("--use_gpu=0")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMatrix)
    unittest.TextTestRunner().run(suite)
    if swig_paddle.isGpuVersion():
        swig_paddle.setUseGpu(True)
        unittest.main()
