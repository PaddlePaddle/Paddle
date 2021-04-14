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
import paddle.fluid as fluid
from paddle.fluid import core

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestNpuCallback(unittest.TestCase):
    def test_static(self):
        # NPU is not supported in ParallelExecutor
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):

            x = paddle.static.data(
                name="x", shape=[4096, 4096], dtype='float32')
            for _ in range(500):
                x = paddle.matmul(x, x)

            t = fluid.LoDTensor()
            if True:
                data = np.ones([1024, 1024])
                t.set(data, paddle.NPUPlace(0))

            copy_data = np.array(t)
            self.assertTrue(np.equal(copy_data, np.ones([1024, 1024])).all())

        x_np = np.random.random([4096, 4096]).astype('float32')
        compiled_prog = paddle.static.CompiledProgram(prog)
        place = paddle.NPUPlace(0)
        exe = paddle.static.Executor(place)

        with self.assertRaisesRegex(RuntimeError,
                                    "NPU is not supported in ParallelExecutor"):
            exe.run(compiled_prog, feed={"x": x_np})


if __name__ == '__main__':
    unittest.main()
