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
"""
https://devblogs.nvidia.com/mixed-precision-training-deep-neural-networks/
Use half-precision to training neural-network with tiny accuracy loss even without accuracy loss.
pros:
1. Shorten the training or inference time
2. Decrease the required amount of memory
cons:
may loss some precision. Please keep an eye open on the convergence result.
"""

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.framework import Program
from paddle.fluid.executor import global_scope
from float16_transpiler import Float16Transpiler

import paddle.fluid.layers as layers
import paddle.fluid.optimizer as optimizer
from paddle.fluid.framework import Program, program_guard
from paddle.fluid.transpiler import memory_optimize
"""
Fp16Training transpiler take a program, will
generate a support fp16 training. Successful training
of DNNs with half precision need three pres:
1.accumulation of FP16 products into FP32;
2.loss scaling;
3.an FP32 master copy of weights for optimize;
"""


class Float16TrainingTranspiler(Float16Transpiler):
    def transpile(self, program, place, scope=None):
        if not isinstance(program, Program):
            raise TypeError("Expect argument of Program, but get %s" %
                            (type(program)))
        if not isinstance(place, core.CPUPlace) and \
                            not isinstance(place, core.CUDAPlace):
            raise TypeError("place should be as CPUPlace/CUDAPlace type")
        if scope is None:
            scope = global_scope()
        if not isinstance(scope, core.Scope):
            raise TypeError("scope should be as Scope type or None")


def GenProgram():
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
    program = GenProgram()
    place = fluid.CPUPlace()
    # t = Float16TrainingTranspiler()
    t = Float16Transpiler()
    t.transpile(program, place)
