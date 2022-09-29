#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard
from paddle.fluid.layers.nn import _pull_gpups_sparse

paddle.enable_static()


class TestPullGpupsSparse(unittest.TestCase):
    """Test PullGpupsSparse op."""

    def test_static_graph(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        slots = []
        with fluid.program_guard(train_program, startup_program):

            l = fluid.layers.data(name='input',
                                  shape=[1],
                                  dtype="int64",
                                  lod_level=1)
            slots.append(l)
            output = _pull_gpups_sparse(slots,
                                        size=[11],
                                        is_distributed=True,
                                        is_sparse=True)
            cost = paddle.mean(output)
            sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
            sgd_optimizer.minimize(cost, train_program)
            block = train_program.global_block()
            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(startup_program)
            img = np.array([1]).astype(np.int64)
            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])


if __name__ == "__main__":
    unittest.main()
