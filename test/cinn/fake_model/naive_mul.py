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

size = 30
paddle.enable_static()

a = static.data(name="A", shape=[-1, size], dtype='float32')
label = static.data(name="label", shape=[size], dtype='float32')

a1 = static.nn.fc(
    x=a, size=size, activation="relu", bias_attr=None, num_flatten_dims=1
)

cost = paddle.nn.functional.square_error_cost(a1, label)
avg_cost = paddle.mean(cost)

optimizer = paddle.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_cost)

cpu = paddle.CPUPlace()
loss = exe = static.Executor(cpu)

exe.run(static.default_startup_program())

static.io.save_inference_model("./naive_mul_model", [a], [a1], exe)
print('res is : ', a1.name)
