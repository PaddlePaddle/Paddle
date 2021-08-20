#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import paddle
import numpy as np
from paddle.fluid import core

paddle.enable_static()


class TestNpuPlace(unittest.TestCase):
    def test(self):
        p = core.Place()
        p.set_place(paddle.NPUPlace(0))

        self.assertTrue(p.is_npu_place())
        self.assertEqual(p.npu_device_id(), 0)


class TestNpuPlaceError(unittest.TestCase):
    def test_static(self):
        # NPU is not supported in ParallelExecutor
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):

            x_np = np.array([2, 3, 4]).astype('float32')
            y_np = np.array([1, 5, 2]).astype('float32')

            x = paddle.static.data(name="x", shape=[3], dtype='float32')
            y = paddle.static.data(name="y", shape=[3], dtype='float32')
            z = paddle.add(x, y)

        compiled_prog = paddle.static.CompiledProgram(prog)
        place = paddle.NPUPlace(0)
        exe = paddle.static.Executor(place)

        with self.assertRaisesRegex(RuntimeError,
                                    "NPU is not supported in ParallelExecutor"):
            exe.run(compiled_prog, feed={"x": x_np, "y": y_np})


if __name__ == '__main__':
    unittest.main()
