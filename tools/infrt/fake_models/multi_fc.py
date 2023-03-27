# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
A fake model with multiple FC layers to test CINN on a more complex model.
"""
import paddle
from paddle import fluid

size = 2
num_layers = 4
paddle.enable_static()

a = paddle.static.data(name="A", shape=[-1, size], dtype='float32')
label = paddle.static.data(name="label", shape=[-1, size], dtype='float32')

fc_out = paddle.static.nn.fc(
    x=a,
    size=size,
    activation="relu",
    bias_attr=fluid.ParamAttr(name="fc_bias"),
    num_flatten_dims=1,
)

for i in range(num_layers - 1):
    fc_out = paddle.static.nn.fc(
        x=fc_out,
        size=size,
        activation="relu",
        bias_attr=fluid.ParamAttr(name="fc_bias"),
        num_flatten_dims=1,
    )

cost = fluid.layers.square_error_cost(fc_out, label)
avg_cost = paddle.mean(cost)

optimizer = fluid.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = fluid.core.CPUPlace()
loss = exe = fluid.Executor(cpu)

exe.run(fluid.default_startup_program())

fluid.io.save_inference_model("./multi_fc_model", [a.name], [fc_out], exe)
fluid.io.save_inference_model(
    "./multi_fc_model",
    [a.name],
    [fc_out],
    exe,
    None,
    "fc.pdmodel",
    "fc.pdiparams",
)

print('output name', fc_out.name)
