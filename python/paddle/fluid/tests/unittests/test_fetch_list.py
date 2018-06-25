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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import op_test
import numpy
import unittest


class TestFetchList(op_test.OpTest):
    def test_fetch_list(self):
        val = numpy.array([1, 3, 5]).astype(numpy.int32)
        x = layers.create_tensor(dtype="int32", persistable=True, name="x")
        y = layers.create_tensor(dtype="int32", name="y")
        print dir(x)
        print type(x)
        layers.assign(input=val, output=x)
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        fluid.memory_optimize(fluid.default_main_program())

        exe.run(fluid.default_main_program(), feed={}, fetch_list=[x])
        try:
            exe.run(fluid.default_main_program(), feed={}, fetch_list=[y])
            self.assertEqual(0, 1)
        except Exception as e:
            pass

        train_exe = fluid.ParallelExecutor(use_cuda=True)
        try:
            loss, = train_exe.run(fetch_list=[y], feed={})
            self.assertEqual(0, 1)
        except Exception as e:
            pass


if __name__ == '__main__':
    unittest.main()
