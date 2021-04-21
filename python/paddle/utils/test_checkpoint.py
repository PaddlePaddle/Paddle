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

paddle.seed(10)
np.random.seed(10)
random.seed(10)


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
    def __init__(self, input_size=10, checkpoint_blocks=[]):
        super(MyModel, self).__init__()
        self.checkpoint_blocks = checkpoint_blocks
        self.runfunc0 = get_fc_block(0, input_size, is_last=False)
        self.runfunc1 = get_fc_block(1, input_size, is_last=False)
        self.runfunc2 = get_fc_block(2, input_size, is_last=True)

    def forward(self, inputs):

        if 0 in self.checkpoint_blocks:
            inputs = checkpoint(self.runfunc0, inputs, preserve_rng_state=False)
        else:
            inputs = self.runfunc0(inputs)

        if 1 in self.checkpoint_blocks:
            inputs = checkpoint(self.runfunc1, inputs, preserve_rng_state=False)
        else:
            inputs = self.runfunc1(inputs)

        if 2 in self.checkpoint_blocks:
            inputs = checkpoint(self.runfunc2, inputs, preserve_rng_state=False)
        else:
            inputs = self.runfunc2(inputs)

        return inputs


def main():
    batch_size, input_size = 1, 10
    model = MyModel(input_size, checkpoint_blocks=[0, 1])
    loss_fn = paddle.nn.MSELoss(reduction='mean')
    optimizer = paddle.optimizer.SGD(learning_rate=0.01,
                                     parameters=model.parameters())

    print("######" * 2 + "before training" + "######" * 2)
    for param in model.parameters():
        print("name: ", param.name)
        print("name: ", param)
        print("grad: ", param._grad_ivar())

    for step in range(10):
        x_data = np.random.randn(batch_size, input_size).astype(np.float32)
        y_data = np.random.randn(batch_size, 1).astype(np.float32)
        x = paddle.to_tensor(x_data)
        y = paddle.to_tensor(y_data)
        # x.stop_gradient = False
        y_pred = model(x)
        # y_pred.stop_gradient = False
        loss = loss_fn(y_pred, y)
        print("######" * 2 + "step [{}]".format(step) + "######" * 2)
        print("y_pred: ", y_pred)
        print("loss: ", loss)
        loss.backward()
        optimizer.step()
        for param in model.parameters():
            print("name: ", param.name)
            print("name: ", param)
            print("grad: ", param._grad_ivar())
        optimizer.clear_grad()


main()
