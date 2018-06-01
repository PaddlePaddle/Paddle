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

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard


class SSAGraph(object):
    def __init__(self, program):
        self._program = program
        self._vars = []
        self._domainant_frointers = defaultdict(list)
        self._ops = []
        self._ssa_vars = defaultdict(list)

    def build_graph(self):
        """
        build ssa graph
        """
        block = self._program.block(0)
        self._find_dominators(block)

    def _insert_phi(self):
        pass

    def _compute_dom_tree(self):
        pass

    def _find_dominators(self, cfg):
        counter = []
        # for var in block:


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
