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

import paddle
import numpy as np
from paddle.utils import checkpoint
import random

paddle.seed(102)
np.random.seed(102)
random.seed(102)


def get_fc_block(block_idx, input_size, is_last=False):
    block_name = "block_" + str(block_idx)
    block = paddle.nn.Sequential(
        (block_name + "_fc_0", paddle.nn.Linear(
            input_size, input_size, bias_attr=False)),
        (block_name + "_fc_1", paddle.nn.Linear(
            input_size, input_size, bias_attr=False)), )
    if is_last:
        block.add_sublayer(
            block_name + "_fc_2",
            paddle.nn.Linear(
                input_size, 1, bias_attr=False))  # add sublayer
    else:
        block.add_sublayer(
            block_name + "_fc_2",
            paddle.nn.Linear(
                input_size, input_size, bias_attr=False))  # add sublayer
    return block


class MyModel(paddle.nn.Layer):
    def __init__(self, input_size=10, enabel_checkpoint=False):
        super(MyModel, self).__init__()
        self.enabel_checkpoint = enabel_checkpoint
        self.run_function0 = get_fc_block(0, input_size)
        self.run_function1 = get_fc_block(1, input_size)
        self.run_function2 = get_fc_block(2, input_size, is_last=True)

    def forward(self, inputs):
        if self.enabel_checkpoint:
            x = self.run_function0(inputs)
            x = checkpoint(self.run_function1, x, preserve_rng_state=False)
            out = self.run_function2(x)
        else:
            x = self.run_function0(inputs)
            x = self.run_function1(x)
            out = self.run_function2(x)

        return out


def main():
    batch_size, input_size = 1, 10

    x_data = np.random.randn(batch_size, input_size).astype(np.float32)
    y_data = np.random.randn(batch_size, 1).astype(np.float32)

    model = MyModel(input_size, enabel_checkpoint=True)
    loss_fn = paddle.nn.MSELoss(reduction='mean')
    optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                     parameters=model.parameters())

    x = paddle.to_tensor(x_data)
    y = paddle.to_tensor(y_data)
    y_pred = model(x)
    # y_pred.stop_gradient = True
    print("y_pred: ", y_pred)
    print("y_pred:", y_pred.stop_gradient)
    loss = loss_fn(y_pred, y)
    print("loss: ", loss)
    print("loss:", loss.stop_gradient)

    loss.backward()
    for param in model.parameters():
        print("name: ", param.name)
        print("grad: ", param._grad_ivar())
    optimizer.step()


main()
