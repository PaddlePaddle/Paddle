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

update_method = "collective"


def net(input):
    return fluid.layers.fc(input, [1000], act="softmax")


def local_optimize(optimizer, params_grads):
    optimizer.apply_gradients(params_grads)


def pserver_optimize(optimizer, params_grads):
    with fluid.placement("127.0.0.1:7164,127.0.0.1:7165"):
        fluid.copy_initializers([pg[0] for pg in params_grads])
        optimizer.apply_gradients(params_grads)


def collective_optimize(optimizer, params_grads):
    new_p_g = []
    for p, g in params_grads:
        sum_grad = fluid.layers.allreduce(g)
        new_p_g.append(p, sum_grad)
    optimizer.apply_gradients(new_p_g)


input = fluid.layers.data("input", [100, 100])
label = fluid.layers.data("label", [1000])
model = net(input)

cost = fluid.layers.cross_entropy(input, label).sum()
optimizer = fluid.optimizer.SGD(0.1)
params_grads = optimizer.backward(cost)

# choose from below optimize method:
if update_method == "local":
    local_optimize(optimizer, params_grads)
elif update_method == "pserver":
    pserver_optimize(optimizer, params_grads)
elif update_method == "collective":
    collective_optimize(optimizer, params_grads)

exe = fluid.ParallelExecutor(True)
exe.replicate(devices=[0, 1, 2, 3])
for step in range(1000):
    exe.run()
