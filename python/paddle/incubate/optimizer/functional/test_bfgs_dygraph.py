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

import numpy as np
import paddle
import paddle.nn.functional as F
from bfgs import minimize_bfgs


class LossAndFlatGradient():
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, net, build_loss):
        self.net = net
        if paddle.in_dynamic_mode():
            self.weights = net.parameters()
        self.build_loss = build_loss

    def __call__(self, weights_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        Args:
           weights_1d: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `weights_1d`.
        """
        # Set the weights

        self.set_flat_weights(weights_1d)
        loss = self.build_loss()
        if paddle.in_dynamic_mode():
            loss.backward()
            grad = self.dynamic_stitch([param.grad for param in self.weights])
            for weight in self.weights:
                weight.clear_grad()

        return loss, grad

    def dynamic_stitch(self, inputs):
        flattened_weights = [paddle.flatten(weight) for weight in inputs]
        concat_weights = paddle.concat(flattened_weights)
        return concat_weights

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.
        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        #weights = self.dynamic_partition(weights_1d, self.partitions,
        #self.n_tensors)
        #for i, (shape, param) in enumerate(zip(self.shapes, weights)):
        #paddle.assign(self.net.parameters()[i], paddle.reshape(param, shape))

        with paddle.no_grad():
            if paddle.in_dynamic_mode():
                for i in range(len(self._flat_weight)):
                    self._flat_weight[i] = weights_1d[i]

    def to_flat_weights(self, weights):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        shape = [np.prod(param.shape) for param in self.weights]

        _all_weights = [None] * len(self.weights)
        _all_weights = [w for w in self.weights]

        self._flat_weight = paddle.create_parameter(
            shape=[np.sum(shape)], dtype=self.weights[0].dtype)
        if paddle.fluid.in_dygraph_mode():
            with paddle.no_grad():
                _C_ops.coalesce_tensor(
                    _all_weights, _all_weights, self._flat_weight, "copy_data",
                    True, "use_align", False, "dtype", self.weights[0].dtype)
        return self._flat_weight


class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.linear_1 = paddle.nn.Linear(1, 8)
        self.linear_2 = paddle.nn.Linear(8, 1)
        self.relu = paddle.nn.ReLU()

    def forward(self, inputs):
        y = self.linear_1(inputs)
        y = self.relu(y)
        y = self.linear_2(y)
        return y


# dygraph

paddle.disable_static()

net = Net()
loss_fn = paddle.nn.L1Loss()

for epoch in range(5):
    x_data = paddle.rand([16]).unsqueeze(1)  # 训练数据
    y_data = x_data + 2.0  # 训练数据标签

    def loss():
        #print("x_data: ", x_data)
        predicts = net(x_data)  # 预测结果
        loss = loss_fn(predicts, y_data)
        return loss

    func = LossAndFlatGradient(net, loss)
    initial_position = func.to_flat_weights(net.parameters())
    results = minimize_bfgs(func, initial_position=initial_position)
    func.set_flat_weights(results[2])

    print("loss: ", results[3])
