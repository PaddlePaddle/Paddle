# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from .framework import Program, program_guard, unique_name, default_startup_program
from .param_attr import ParamAttr
from .initializer import Constant
from . import layers
from . import backward
from .dygraph import Layer, nn
from . import executor

from . import core
import numpy as np

__all__ = ['run_check']


class SimpleLayer(Layer):
    def __init__(self, name_scope):
        super(SimpleLayer, self).__init__(name_scope)
        self._fc1 = nn.FC(self.full_name(),
                          3,
                          param_attr=ParamAttr(initializer=Constant(value=0.1)))

    def forward(self, inputs):
        x = self._fc1(inputs)
        x = layers.reduce_sum(x)
        return x


def run_check():
    ''' intall check to verify if install is success

    This func should not be called only if you need to verify installation
    '''
    print("Running Verify Fluid Program ... ")
    prog = Program()
    startup_prog = Program()
    scope = core.Scope()
    with executor.scope_guard(scope):
        with program_guard(prog, startup_prog):
            with unique_name.guard():
                np_inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
                inp = layers.data(
                    name="inp", shape=[2, 2], append_batch_size=False)
                simple_layer = SimpleLayer("simple_layer")
                out = simple_layer(inp)
                param_grads = backward.append_backward(
                    out, parameter_list=[simple_layer._fc1._w.name])[0]
                exe = executor.Executor(core.CPUPlace(
                ) if not core.is_compiled_with_cuda() else core.CUDAPlace(0))
                exe.run(default_startup_program())
                exe.run(feed={inp.name: np_inp},
                        fetch_list=[out.name, param_grads[1].name])

    print(
        "Your Paddle Fluid is installed successfully! Let's start deep Learning with Paddle Fluid now"
    )
