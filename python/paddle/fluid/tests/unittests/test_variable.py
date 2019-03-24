#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
from paddle.fluid.framework import default_main_program, Program, convert_np_dtype_to_dtype_
import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np
from test_imperative_base import new_program_scope


class TestVariable(unittest.TestCase):
    def test_np_dtype_convert(self):
        DT = core.VarDesc.VarType
        convert = convert_np_dtype_to_dtype_
        self.assertEqual(DT.FP32, convert(np.float32))
        self.assertEqual(DT.FP16, convert("float16"))
        self.assertEqual(DT.FP64, convert("float64"))
        self.assertEqual(DT.INT32, convert("int32"))
        self.assertEqual(DT.INT16, convert("int16"))
        self.assertEqual(DT.INT64, convert("int64"))
        self.assertEqual(DT.BOOL, convert("bool"))
        self.assertEqual(DT.INT8, convert("int8"))
        self.assertEqual(DT.UINT8, convert("uint8"))

    def test_var(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100], lod_level=0, name="fc.w")
        self.assertNotEqual(str(w), "")
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual(0, w.lod_level)

        w = b.create_var(name='fc.w')
        self.assertEqual(core.VarDesc.VarType.FP64, w.dtype)
        self.assertEqual((784, 100), w.shape)
        self.assertEqual("fc.w", w.name)
        self.assertEqual(0, w.lod_level)

        self.assertRaises(ValueError,
                          lambda: b.create_var(name="fc.w", shape=(24, 100)))

    def test_step_scopes(self):
        prog = Program()
        b = prog.current_block()
        var = b.create_var(
            name='step_scopes', type=core.VarDesc.VarType.STEP_SCOPES)
        self.assertEqual(core.VarDesc.VarType.STEP_SCOPES, var.type)

    def _test_slice(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100, 100], lod_level=0)

        for i in range(3):
            nw = w[i]
            self.assertEqual((1, 100, 100), nw.shape)

        nw = w[:]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[:, :, :]
        self.assertEqual((784, 100, 100), nw.shape)

        nw = w[::2, ::2, :]
        self.assertEqual((392, 50, 100), nw.shape)

        nw = w[::-2, ::-2, :]
        self.assertEqual((392, 50, 100), nw.shape)

        self.assertEqual(0, nw.lod_level)

    def test_slice(self):
        self._test_slice()

class TestVariableImperative(unittest.TestCase):
    def _test_slice(self):
        b = default_main_program().current_block()
        w = b.create_var(
            dtype="float64", shape=[784, 100, 100], lod_level=0)

        for i in range(3):
            nw = w[i]
            self.assertEqual([1, 100, 100], nw.shape)

        nw = w[:]
        self.assertEqual([784, 100, 100], nw.shape)

        nw = w[:, :, :]
        self.assertEqual([784, 100, 100], nw.shape)

        nw = w[::2, ::2, :]
        self.assertEqual([392, 50, 100], nw.shape)

        nw = w[::-2, ::-2, :]
        self.assertEqual([392, 50, 100], nw.shape)

        nw = w[0::-2, 0::-2, :]
        self.assertEqual([1, 1, 100], nw.shape)

    def test_slice(self):
        with fluid.imperative.guard():
            self._test_slice()

if __name__ == '__main__':
    unittest.main()
