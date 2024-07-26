# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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


import paddle
from paddle import static

paddle.enable_static()

resnet_input = static.data(
    name="resnet_input", shape=[1, 160, 7, 7], dtype='float32'
)
label = static.data(name="label", shape=[1, 960, 7, 7], dtype='float32')
d = paddle.nn.functional.relu6(resnet_input)
f = paddle.nn.Conv2D(
    in_channels=d.shape[1],
    out_channels=960,
    kernel_size=1,
    stride=1,
    dilation=1,
    padding=0,
)(d)

g = paddle.nn.Conv2D(
    in_channels=f.shape[1],
    out_channels=160,
    kernel_size=1,
    stride=1,
    dilation=1,
    padding=0,
)(f)
i = paddle.nn.Conv2D(
    in_channels=g.shape[1],
    out_channels=960,
    kernel_size=1,
    stride=1,
    dilation=1,
    padding=0,
)(g)
j1 = paddle.scale(i, scale=2.0, bias=0.5)
j = paddle.scale(j1, scale=2.0, bias=0.5)
temp7 = paddle.nn.functional.relu(j)

cost = paddle.nn.functional.square_error_cost(temp7, label)
avg_cost = paddle.mean(cost)

optimizer = paddle.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = paddle.CPUPlace()
exe = static.Executor(cpu)

exe.run(static.default_startup_program())

static.io.save_inference_model("./resnet_model", [resnet_input], [temp7], exe)
static.io.save_inference_model("./resnet_model_1", [resnet_input], [temp7], exe)
if paddle.framework.use_pir_api():
    print('res', temp7)
else:
    print('res', temp7.name)
