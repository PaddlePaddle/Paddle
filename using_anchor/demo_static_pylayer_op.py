# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np

import paddle
from paddle import static

paddle.enable_static()


def forward_fn(x):
    y = paddle.tanh(x)
    return y


def backward_fn(dy):
    dx = dy * 2
    return dx


train_program = static.Program()
start_program = static.Program()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
with static.program_guard(train_program, start_program):
    data = paddle.static.data(name="X", shape=[None, 5], dtype="float32")
    data.stop_gradient = False
    ret = paddle.static.nn.static_pylayer(forward_fn, [data], backward_fn)
    loss = paddle.mean(ret)
    sgd_opt = paddle.optimizer.SGD(learning_rate=0.01).minimize(loss)
    print(static.default_main_program())

exe = paddle.static.Executor(place)
exe.run(start_program)
x = np.random.randn(10, 5).astype(np.float32)
loss, loss_g, x_g, y, y_g = exe.run(
    train_program,
    feed={"X": x},
    fetch_list=[
        loss.name,
        loss.name + '@GRAD',
        data.name + '@GRAD',
        ret.name,
        ret.name + '@GRAD',
    ],
)
print("x = ")
print(x)
print("x_g = ")
print(x_g)
print("loss = ")
print(loss)
print("loss_g = ")
print(loss_g)
print("y = ")
print(y)
print("y_g = ")
print(y_g)

# to validate
numpy_ret = np.mean(x)
print("numpy_ret = ")
print(numpy_ret)

np.allclose(y, numpy_ret)
