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

# static
paddle.enable_static()


class LossAndFlatGradient():
    """A helper class to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        trainable_variables: Trainable variables.
        build_loss: A function to build the loss function expression.
    """

    def __init__(self, build_loss):
        if not paddle.in_dynamic_mode():
            print(paddle.fluid.framework.default_main_program())
            params = paddle.fluid.framework.default_main_program().global_block(
            ).all_parameters()
            parameters = [param.name for param in params if param.trainable]
            self.weights = []
            self.shapes = [param.shape for param in params if param.trainable]
            for param in parameters:
                param_var = paddle.fluid.framework.default_main_program(
                ).global_block().var(param)
                self.weights.append(param_var)

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
        with paddle.fluid.program_guard(
                paddle.fluid.framework.default_main_program(),
                paddle.fluid.framework.default_startup_program()):
            params_grads = paddle.static.append_backward(loss)

        grad = self.dynamic_stitch([pair[1].detach() for pair in params_grads])
        print("grad: ", grad)
        return loss, grad

    def dynamic_stitch(self, inputs=None):
        print("inputs: ", inputs)
        if inputs is None:
            inputs = self.weights
        with paddle.no_grad():
            print("inputs[0]: ", inputs[0])
            paddle.fluid.layers.flatten(inputs[0], 0)
            flattened_weights = [
                paddle.fluid.layers.squeeze(
                    paddle.fluid.layers.flatten(weight, 0), [0])
                for weight in inputs
            ]
            concat_weights = paddle.concat(flattened_weights)
        return concat_weights

    def set_flat_weights(self, weights_1d):
        """Sets the weights with a 1D tf.Tensor.
        Args:
            weights_1d: a 1D tf.Tensor representing the trainable variables.
        """
        with paddle.no_grad():
            sections = [np.prod(list(shape)) for shape in self.shapes]
            weights = paddle.split(weights_1d, sections)

        #for i, (shape, param) in enumerate(zip(self.shapes, weights)):
        #paddle.assign(self.net.parameters()[i], paddle.reshape(param, shape))

        # with paddle.no_grad():
        #     for i in range(len(self.weights)):
        #         paddle.assign(weights_1d[i], self.weights[])

    def to_flat_weights(self):
        """Returns a 1D tf.Tensor representing the `weights`.
        Args:
            weights: A list of tf.Tensor representing the weights.
        Returns:
            A 1D tf.Tensor representing the `weights`.
        """
        shape = [
            paddle.prod(paddle.fluid.layers.shape(param))
            for param in self.weights
        ]
        print("shape: ", shape)
        _all_weights = [None] * len(self.weights)
        _all_weights = [w for w in self.weights]

        self._flat_weight = paddle.create_parameter(
            shape=[paddle.fluid.layers.sum(shape)], dtype='float32')
        if not paddle.fluid.in_dygraph_mode():
            with paddle.no_grad():
                _C_ops.coalesce_tensor(
                    _all_weights, _all_weights, self._flat_weight, "copy_data",
                    True, "use_align", False, "dtype", self.weights[0].dtype)
        return self._flat_weight


def StaticNet():
    X = paddle.static.data(name='x', shape=[1, 1], dtype=dtype)
    x1 = paddle.static.nn.fc(x=X, size=8)
    x2 = paddle.static.nn.fc(x=x1, size=1)
    x3 = paddle.fluid.layers.relu(x=x2)
    return x3


def loss_fn():
    y_predict = StaticNet()
    Y = paddle.static.data(name='y', shape=[1, 1], dtype=dtype)
    loss = paddle.fluid.layers.mse_loss(y_predict, Y)
    return loss


dtype = 'float32'
loss = loss_fn()
print("loss: ", loss)
func = LossAndFlatGradient(lambda: loss_fn())
initial_position = func.dynamic_stitch()
bfgs_op = minimize_bfgs(func, initial_position=initial_position)

exe = paddle.static.Executor()
exe.run(paddle.fluid.framework.default_startup_program())

for epoch in range(5):
    x_data = np.rand([1])  # 训练数据
    y_data = x_data + 2.0  # 训练数据标签
    results = exe.run(bfgs_op, feed={'x': x_data, 'y': y_data})
