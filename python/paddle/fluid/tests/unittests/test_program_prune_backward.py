#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import contextlib
import numpy
import paddle.fluid as fluid
import paddle.fluid.core as core


class TestProgramPruneBackward(unittest.TestCase):
    def test_simple_fc_network(self):
        with self.program_scope_guard():
            x = fluid.layers.data(name='X', shape=[13], dtype='float32')
            y = fluid.layers.data(name='Y', shape=[1], dtype='float32')
            y_ = fluid.layers.fc(input=x, size=1, act=None)
            loss = fluid.layers.square_error_cost(input=y_, label=y)
            avg_loss = fluid.layers.mean(loss)

            main_program = fluid.default_main_program()
            test_program_orig = main_program.clone(for_test=True)
            fluid.optimizer.SGD(learning_rate=0.01).minimize(avg_loss)
            test_program_prune = main_program.clone(for_test=True)

            place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())

            input_x = numpy.random.random(size=(10, 13)).astype('float32')
            input_y = numpy.random.random(size=(10, 1)).astype('float32')
            loss_data_in_orig, = exe.run(test_program_orig,
                                         feed={'X': input_x,
                                               'Y': input_y},
                                         fetch_list=[avg_loss.name])
            loss_data_in_prune, = exe.run(test_program_prune,
                                          feed={'X': input_x,
                                                'Y': input_y},
                                          fetch_list=[avg_loss.name])

            self.assertEqual(loss_data_in_orig, loss_data_in_prune)

    @contextlib.contextmanager
    def program_scope_guard(self):
        prog = fluid.Program()
        startup_prog = fluid.Program()
        scope = fluid.core.Scope()
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                yield


if __name__ == '__main__':
    unittest.main()
