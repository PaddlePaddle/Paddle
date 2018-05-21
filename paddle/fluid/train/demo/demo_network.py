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
import paddle.fluid.framework as framework


def train_network(with_optimize):
    x = fluid.layers.data(name='x', shape=[13], dtype='float32')
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    y = fluid.layers.data(name='y', shape=[1], dtype='float32')
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_cost = fluid.layers.mean(cost)

    if with_optimize:
        sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.00001)
        sgd_optimizer.minimize(avg_cost)
    else:
        fluid.backward.append_backward(avg_cost)


def save_program_desc(network_func):
    startup_program = framework.Program()
    train_program = framework.Program()

    with framework.program_guard(train_program, startup_program):
        network_func(with_optimize=False)

    with open("startup_program", "w") as f:
        f.write(startup_program.desc.serialize_to_string())
    with open("main_program", "w") as f:
        f.write(train_program.desc.serialize_to_string())


save_program_desc(train_network)
