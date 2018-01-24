#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

import paddle.v2 as paddle
import paddle.v2.fluid as fluid
import paddle.v2.fluid.core as core
import paddle.v2.fluid.layers as layers
import paddle.v2.fluid.framework as framework
import numpy as np
from paddle.v2.fluid.backward import append_backward

label_dim = 5
dim = 5

x = fluid.layers.data(name='x', shape=[dim], dtype='float32', lod_level=1)

t = fluid.layers.data(name='t', shape=[1], dtype='float32', lod_level=1)

rnn = fluid.layers.DynamicRNN()

with rnn.block():
    step_t = rnn.step_input(t)
    step_x = rnn.step_input(x)
    out_mem = rnn.memory(value=0, dtype='float32', shape=[dim])
    out_mem.stop_gradient = True
    value, idx = layers.topk(out_mem, 1)
    mul = layers.elementwise_mul(x=step_t, y=layers.cast(idx, 'float32'))
    rnn.update_memory(out_mem, step_x)
    rnn.output(mul)

decoder_output = rnn()
loss = layers.mean(x=decoder_output)
append_backward(loss=loss)


def lodtensor_to_ndarray(lod_tensor):
    dims = lod_tensor.get_dims()
    ndarray = np.zeros(shape=dims).astype('float32')
    for i in xrange(np.product(dims)):
        ndarray.ravel()[i] = lod_tensor.get_float_element(i)
    return ndarray, lod_tensor.lod()


place = core.CPUPlace()
lod = [[0, 1, 3]]
x_shape = [lod[0][-1], dim]
x_data = np.random.random(x_shape).astype('float32')
x_tensor = core.LoDTensor()
x_tensor.set_lod(lod)
x_tensor.set(x_data, place)

shape = [[lod[0][-1], 1]]
t_data = [np.random.randint(0, label_dim - 1) for i in xrange(lod[0][-1])]
t_data = np.array(t_data).astype('float32').reshape((-1, 1))
t_tensor = core.LoDTensor()
t_tensor.set_lod(lod)
t_tensor.set(t_data, place)

exe = fluid.Executor(place)
print(framework.default_main_program().block(2))
exe.run(framework.default_startup_program())

fetch_outs = exe.run(feed={'x': x_tensor,
                           't': t_tensor},
                     fetch_list=[loss],
                     return_numpy=False)

out, lod = lodtensor_to_ndarray(fetch_outs[0])

print out, lod
