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
import util
import numpy as np
import unittest


class TestIVector(unittest.TestCase):
    def test_createZero(self):
        m = swig_paddle.IVector.createZero(10, False)
        self.assertIsNotNone(m)
        for i in xrange(10):
            self.assertEqual(m[i], 0)
            m[i] = i
            self.assertEqual(m[i], i)

        m = swig_paddle.IVector.createZero(10)
        self.assertEqual(m.isGpu(), swig_paddle.isUsingGpu())
        self.assertEqual(m.getData(), [0] * 10)

    def test_create(self):
        m = swig_paddle.IVector.create(range(10), False)
        self.assertIsNotNone(m)
        for i in xrange(10):
            self.assertEqual(m[i], i)

        m = swig_paddle.IVector.create(range(10))
        self.assertEqual(m.isGpu(), swig_paddle.isUsingGpu())
        self.assertEqual(m.getData(), range(10))

    def test_cpu_numpy(self):
        vec = np.array([1, 3, 4, 65, 78, 1, 4], dtype="int32")
        iv = swig_paddle.IVector.createCpuVectorFromNumpy(vec, False)
        self.assertEqual(vec.shape[0], int(iv.__len__()))
        vec[4] = 832
        for i in xrange(len(iv)):
            self.assertEqual(vec[i], iv[i])
        vec2 = iv.toNumpyArrayInplace()
        vec2[1] = 384
        for i in xrange(len(iv)):
            self.assertEqual(vec[i], iv[i])
            self.assertEqual(vec2[i], iv[i])

    def test_gpu_numpy(self):
        if swig_paddle.isGpuVersion():
            vec = swig_paddle.IVector.create(range(0, 10), True)
            assert isinstance(vec, swig_paddle.IVector)
            self.assertTrue(vec.isGpu())
            self.assertEqual(vec.getData(), range(0, 10))
            num_arr = vec.copyToNumpyArray()
            assert isinstance(num_arr, np.ndarray)  # for code hint.
            num_arr[4] = 7
            self.assertEquals(vec.getData(), range(0, 10))

            vec.copyFromNumpyArray(num_arr)
            expect_vec = range(0, 10)
            expect_vec[4] = 7
            self.assertEqual(vec.getData(), expect_vec)

    def test_numpy(self):
        vec = np.array([1, 3, 4, 65, 78, 1, 4], dtype="int32")
        iv = swig_paddle.IVector.createVectorFromNumpy(vec)
        self.assertEqual(iv.isGpu(), swig_paddle.isUsingGpu())
        self.assertEqual(iv.getData(), list(vec))


class TestVector(unittest.TestCase):
    def testCreateZero(self):
        v = swig_paddle.Vector.createZero(10, False)
        self.assertIsNotNone(v)
        for i in xrange(len(v)):
            self.assertTrue(util.doubleEqual(v[i], 0))
            v[i] = i
            self.assertTrue(util.doubleEqual(v[i], i))

        v = swig_paddle.Vector.createZero(10)
        self.assertEqual(v.isGpu(), swig_paddle.isUsingGpu())
        self.assertEqual(v.getData(), [0] * 10)

    def testCreate(self):
        v = swig_paddle.Vector.create([x / 100.0 for x in xrange(100)], False)
        self.assertIsNotNone(v)
        for i in xrange(len(v)):
            self.assertTrue(util.doubleEqual(v[i], i / 100.0))
        self.assertEqual(100, len(v))

        v = swig_paddle.Vector.create([x / 100.0 for x in xrange(100)])
        self.assertEqual(v.isGpu(), swig_paddle.isUsingGpu())
        self.assertEqual(100, len(v))
        vdata = v.getData()
        for i in xrange(len(v)):
            self.assertTrue(util.doubleEqual(vdata[i], i / 100.0))

    def testCpuNumpy(self):
        numpy_arr = np.array([1.2, 2.3, 3.4, 4.5], dtype="float32")
        vec = swig_paddle.Vector.createCpuVectorFromNumpy(numpy_arr, False)
        assert isinstance(vec, swig_paddle.Vector)
        numpy_arr[0] = 0.1
        for n, v in zip(numpy_arr, vec):
            self.assertTrue(util.doubleEqual(n, v))

        numpy_2 = vec.toNumpyArrayInplace()
        vec[0] = 1.3
        for x, y in zip(numpy_arr, numpy_2):
            self.assertTrue(util.doubleEqual(x, y))

        for x, y in zip(numpy_arr, vec):
            self.assertTrue(util.doubleEqual(x, y))

        numpy_3 = vec.copyToNumpyArray()
        numpy_3[0] = 0.4
        self.assertTrue(util.doubleEqual(vec[0], 1.3))
        self.assertTrue(util.doubleEqual(numpy_3[0], 0.4))

        for i in xrange(1, len(numpy_3)):
            util.doubleEqual(numpy_3[i], vec[i])

    def testNumpy(self):
        numpy_arr = np.array([1.2, 2.3, 3.4, 4.5], dtype="float32")
        vec = swig_paddle.Vector.createVectorFromNumpy(numpy_arr)
        self.assertEqual(vec.isGpu(), swig_paddle.isUsingGpu())
        vecData = vec.getData()
        for n, v in zip(numpy_arr, vecData):
            self.assertTrue(util.doubleEqual(n, v))

    def testCopyFromNumpy(self):
        vec = swig_paddle.Vector.createZero(1, False)
        arr = np.array([1.3, 3.2, 2.4], dtype="float32")
        vec.copyFromNumpyArray(arr)
        for i in xrange(len(vec)):
            self.assertTrue(util.doubleEqual(vec[i], arr[i]))


if __name__ == '__main__':
    swig_paddle.initPaddle("--use_gpu=0")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVector)
    unittest.TextTestRunner().run(suite)
    if swig_paddle.isGpuVersion():
        swig_paddle.setUseGpu(True)
        unittest.main()
