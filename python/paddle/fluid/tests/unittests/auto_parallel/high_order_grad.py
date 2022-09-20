# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import random
import paddle
import unittest
import numpy as np
from paddle.distributed.fleet import auto
from paddle.incubate.autograd import Hessian

np.random.seed(1234)
paddle.seed(1234)


class FCNet:

    def __init__(self, num_ins, num_outs, num_layers, hidden_size):
        self.num_ins = num_ins
        self.num_outs = num_outs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = paddle.tanh

        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            if i == 0:
                lsize = self.num_ins
                rsize = self.hidden_size
            elif i == (self.num_layers - 1):
                lsize = self.hidden_size
                rsize = self.num_outs
            else:
                lsize = self.hidden_size
                rsize = self.hidden_size

            w = paddle.static.create_parameter(shape=[lsize, rsize],
                                               dtype="float32",
                                               is_bias=False)
            b = paddle.static.create_parameter(shape=[rsize],
                                               dtype="float32",
                                               is_bias=True)
            self.weights.append(w)
            self.biases.append(b)

    def nn_func(self, ins):
        u = ins
        for i in range(self.num_layers - 1):
            u = paddle.nn.functional.linear(u, self.weights[i], self.biases[i])
            u = self.activation(u)
        u = paddle.nn.functional.linear(u, self.weights[-1], self.biases[-1])
        return u


class LaplaceModel(paddle.nn.Layer):

    def __init__(self, num_ins=2, num_outs=1, num_layers=5, hidden_size=20):
        super(LaplaceModel, self).__init__()
        self.net = FCNet(num_ins=num_ins,
                         num_outs=num_outs,
                         num_layers=num_layers,
                         hidden_size=hidden_size)

    def forward(self, inputs, bc_index):
        inputs.stop_gradient = False
        outputs = self.net.nn_func(inputs)
        # eq_loss
        hes = Hessian(self.net.nn_func, inputs, is_batched=True)
        eq_loss = paddle.norm(hes[:, 0, 0] + hes[:, 1, 1], p=2)
        # bc_loss
        bc_u = paddle.index_select(outputs, bc_index)
        return eq_loss, bc_u


class LaplaceDataset(paddle.io.Dataset):

    def __init__(self, num_sample):
        self.num_sample = num_sample

    def __getitem__(self, index):
        x = np.linspace(0, 0.9, 10)
        y = np.linspace(0, 0.9, 10)
        bc_value = np.random.rand(36).reshape(36, 1).astype('float32')

        domain_space = []
        bc_index = []
        for j in range(len(y)):
            for i in range(len(x)):
                domain_space.append([x[i], y[j]])
                if i == 0 or i == 9 or j == 0 or j == 9:
                    bc_index.append(i + 10 * j)
        domain_space = np.array(domain_space, dtype='float32')
        bc_index = np.array(bc_index, dtype='int64')

        return domain_space, bc_index, bc_value

    def __len__(self):
        return self.num_sample


def loss_func(eq_loss, bc_u, bc_value):
    bc_diff = bc_u - bc_value
    bc_loss = paddle.norm(bc_diff, p=2)
    loss = eq_loss + bc_loss
    return loss


def main():
    paddle.enable_static()
    # dataset
    train_dataset = LaplaceDataset(10)
    # optimizer
    optimizer = paddle.optimizer.Adam(learning_rate=0.001)
    # model
    laplace = LaplaceModel()

    dist_strategy = auto.Strategy()
    dist_strategy.auto_mode = "semi"

    engine = auto.Engine(laplace,
                         loss=loss_func,
                         optimizer=optimizer,
                         strategy=dist_strategy)
    engine.fit(train_dataset, train_sample_split=2, batch_size=None)

    dist_context = engine.dist_context
    block = engine.main_program.global_block()
    ops = block.ops
    for op in ops:
        if op.type == 'p_norm':
            op_dist_attr = dist_context.get_op_dist_attr_for_program(op)
            assert op_dist_attr.impl_type == 'p_norm'
        if 'x' in op.input_arg_names:
            out_name = op.output_arg_names[0]
            assert block.vars[out_name].shape[0] == 50


if __name__ == "__main__":
    main()
