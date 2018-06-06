# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from collections import defaultdict
import collections

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard


class VarHandle(object):
    def __init__(self, **kwargs):
        self._version = kwargs.get("version", -1)
        self._name = kwargs.get("name", "")
        self._generated_op = kwargs.get("op", None)
        self._scope_idx = kwargs.get("scope", -1)
        self._place = kwargs.get("place", None)


class OpHandle(object):
    def __init__(self):
        self._inputs = []
        self._outputs = []


class SSAGraph(object):
    def __init__(self, block):
        self._block = block
        self._vars = []
        self._ops = []

    def build_graph(self):
        """
        build ssa graph
        """


def GenTestProgram():
    program = Program()
    with program_guard(program, startup_program=Program()):
        x = layers.data(name='x', shape=[13], dtype='float32')
        y_predict = layers.fc(input=x, size=1, act=None)
        y = layers.data(name='y', shape=[1], dtype='float32')
        cost = layers.square_error_cost(input=y_predict, label=y)
        avg_cost = layers.mean(cost)
        opt = optimizer.SGD(learning_rate=0.001)
        opt = opt.minimize(avg_cost)
    return program


if __name__ == "__main__":
    program = GenTestProgram()
    print("before optimization")
    print(str(program))
    # result_program = memory_optimize(program)
    # print("after optimization")
    # print(str(program))
