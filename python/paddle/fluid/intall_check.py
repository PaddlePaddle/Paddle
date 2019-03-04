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

import paddle.fluid as fluid
import paddle.fluid.core as core
import numpy as np

__all__ = ['install_check']


class SimpleLayer(fluid.imperative.Layer):
    def __init__(self, name_scope):
        super(SimpleLayer, self).__init__(name_scope)

    def forward(self, inputs):
        x = fluid.layers.relu(inputs)
        x = fluid.layers.elementwise_mul(x, x)
        x = fluid.layers.reduce_sum(x)
        return [x]


def install_check():
    ''' intall

    :return:
    '''
    prog = fluid.Program()
    startup_prog = fluid.Program()
    scope = fluid.core.Scope()
    try:
        with fluid.scope_guard(scope):
            with fluid.program_guard(prog, startup_prog):
                with fluid.unique_name.guard():
                    np_inp = np.array([1.0, 2.0, -1.0], dtype=np.float32)
                    inp = fluid.layers.data(
                        name="inp", shape=[2, 2], append_batch_size=False)
                    simple_layer = SimpleLayer("simple_layer")
                    out = simple_layer(inp)
                    param_grads = fluid.backward.append_backward(
                        out, parameter_list=[simple_layer._fc1._w.name])[0]
                    exe = fluid.Executor(core.CPUPlace()
                                         if not core.is_compiled_with_cuda()
                                         else core.CUDAPlace(0))
                    exe.run(fluid.default_startup_program())
                    exe.run(feed={inp.name: np_inp},
                            fetch_list=[out.name, param_grads[1].name])
    except:
        print("Install Failed !")
    else:
        print("Your fluid is installed successfully")
