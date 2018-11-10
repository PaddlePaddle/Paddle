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


class TestArguments(unittest.TestCase):
    def test_load_arguments(self):
        m = swig_paddle.Matrix.createDense([4, 2, 4, 3, 9, 5], 2, 3)
        args = swig_paddle.Arguments.createArguments(1)
        args.setSlotValue(0, m)

        self.assertAlmostEqual(27.0, args.sum())

        mat = args.getSlotValue(0)
        assert isinstance(mat, swig_paddle.Matrix)
        np_mat = mat.toNumpyMatInplace()
        # The matrix unittest is in testMatrix.py
        self.assertEqual(np_mat.shape, (2, 3))

        args.setSlotIds(0, swig_paddle.IVector.create([1, 2, 3, 4, 5, 6]))
        iv = args.getSlotIds(0)
        assert isinstance(iv, swig_paddle.IVector)
        np_arr = iv.toNumpyArrayInplace()
        self.assertEqual(np_arr.shape, (6, ))

    def test_arguments_shape(self):
        h, w = 4, 6
        v = np.random.rand(2, h * w)
        m = swig_paddle.Matrix.createDense(v.flatten(), 2, h * w)
        args = swig_paddle.Arguments.createArguments(1)
        args.setSlotValue(0, m)
        args.setSlotFrameHeight(0, h)
        args.setSlotFrameWidth(0, w)
        self.assertEqual(args.getSlotFrameHeight(), h)
        self.assertEqual(args.getSlotFrameWidth(), w)


if __name__ == '__main__':
    swig_paddle.initPaddle("--use_gpu=0")
    unittest.main()
