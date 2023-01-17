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

import contextlib
import unittest

import paddle
import paddle.fluid as fluid


def train_simulator(test_batch_size=10):
    if test_batch_size <= 0:
        raise ValueError(
            "batch_size should be a positive integeral value, "
            "but got batch_size={}".format(test_batch_size)
        )

    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)
    y = fluid.layers.data(name='y', shape=[1], dtype='float32')

    cost = paddle.nn.functional.square_error_cost(input=y_predict, label=y)
    avg_cost = paddle.mean(cost)

    sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    sgd_optimizer.minimize(avg_cost)

    # Calculate memory usage in current network config
    lower_usage, upper_usage, unit = fluid.contrib.memory_usage(
        fluid.default_main_program(), batch_size=test_batch_size
    )

    print(
        "memory usage is about %.3f - %.3f %s"
        % (lower_usage, upper_usage, unit)
    )


class TestMemoryUsage(unittest.TestCase):
    def test_with_unit_B(self):
        with self.program_scope_guard():
            train_simulator()

    def test_with_unit_KB(self):
        with self.program_scope_guard():
            train_simulator(test_batch_size=1000)

    def test_with_unit_MB(self):
        with self.program_scope_guard():
            train_simulator(test_batch_size=100000)

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
