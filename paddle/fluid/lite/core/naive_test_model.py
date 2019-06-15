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

import numpy
import sys, os
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.backward import append_backward

a = fluid.layers.data(name="a", shape=[2], dtype='float32')
label = fluid.layers.data(name="label", shape=[10], dtype='float32')

a1 = fluid.layers.fc(input=a, size=3, act=None, bias_attr=False)

cost = fluid.layers.square_error_cost(a1, label)
avg_cost = fluid.layers.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())
with open('startup_program.pb', 'wb') as f:
    f.write(fluid.default_startup_program().desc.serialize_to_string())

#data_1 = np.array(numpy.random.random([100, 100]), dtype='float32')

#fluid.default_main_program().desc.

#prog = fluid.compiler.CompiledProgram(fluid.default_main_program())
prog = fluid.default_main_program()

#append_backward(loss)

with open('main_program.pb', 'wb') as f:
    f.write(prog.desc.serialize_to_string())

#outs = exe.run(program=prog, feed={'a':data_1, }, fetch_list=[cost])

#sys.exit(0)
fluid.io.save_inference_model("./model2", [a.name], [a1], exe)

#print(numpy.array(outs))
