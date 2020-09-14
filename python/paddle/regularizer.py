#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

__all__ = ['L1Decay', 'L2Decay']

import paddle.fluid as fluid


class L1Decay(fluid.regularizer.L1Decay):
    """
    Implement the L1 Weight Decay Regularization, which encourages the weights to be sparse.
    
    It can be set in :ref:`api_fluid_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` .
    
    In the implementation, the formula of L1 Weight Decay Regularization is as follows:
	
    .. math::

        L1WeightDecay = reg\_coeff * sign(parameter)

    Args:
        coeff(float, optional): regularization coeff. Default:0.0.
	
    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle
            from paddle.regularizer import L1Decay
            from paddle.vision.models import LeNet
            paddle.disable_static()
            train_dataset = paddle.vision.datasets.MNIST(mode='train')
            test_dataset = paddle.vision.datasets.MNIST(mode='test')
            train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
            def train():
                net = LeNet()
                epochs = 1
                adam = paddle.optimizer.Momentum(learning_rate=0.01,
                                             parameters=net.parameters(),
                                             weight_decay=L1Decay(0.001))
                for epoch in range(epochs):
                    for batch_id, data in enumerate(train_loader()):
                        x_data = data[0]
                        y_data = data[1]
                        predicts = net(x_data)
                        loss = paddle.nn.functional.cross_entropy(predicts, y_data)
                        # calc loss
                        acc = paddle.metric.accuracy(predicts, y_data, k=2)
                        avg_loss = paddle.mean(loss)
                        avg_acc = paddle.mean(acc)
                        avg_loss.backward()
                        if batch_id % 10 == 0:
                            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                        adam.step()
                        adam.clear_grad()
            train()

            # Example2: set Regularizer in parameters
            # Set L1 regularization in parameters.
            # Global regularizer does not take effect on my_conv2d for this case.
            from paddle.nn import Conv2d
            from paddle import ParamAttr
            from paddle.regularizer import L2Decay

            my_conv2d = Conv2d(
                    in_channels=10,
                    out_channels=10,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.0)),
                    bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        super(L1Decay, self).__init__(coeff)


class L2Decay(fluid.regularizer.L2Decay):
    """
    Implement the L2 Weight Decay Regularization, which encourages the weights to be sparse.
    
    It can be set in :ref:`api_fluid_ParamAttr` or ``optimizer`` (such as :ref:`api_paddle_optimizer_Momentum` ). 
    When set in ``ParamAttr`` , it only takes effect for trainable parameters in this layer. When set in 
    ``optimizer`` , it takes effect for all trainable parameters. When set together, ``ParamAttr`` has 
    higher priority than ``optimizer`` .
    
    In the implementation, the formula of L2 Weight Decay Regularization is as follows:

    .. math::

        L2WeightDecay = reg\_coeff * parameter

    Args:
        regularization_coeff(float, optional): regularization coeff. Default:0.0
	
    Examples:
        .. code-block:: python

            # Example1: set Regularizer in optimizer
            import paddle
            from paddle.regularizer import L2Decay
            from paddle.vision.models import LeNet
            paddle.disable_static()
            train_dataset = paddle.vision.datasets.MNIST(mode='train')
            test_dataset = paddle.vision.datasets.MNIST(mode='test')
            train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=64, shuffle=True)
            def train():
                net = LeNet()
                epochs = 1
                adam = paddle.optimizer.Momentum(learning_rate=0.01,
                                             parameters=net.parameters(),
                                             weight_decay=L2Decay(0.001))
                for epoch in range(epochs):
                    for batch_id, data in enumerate(train_loader()):
                        x_data = data[0]
                        y_data = data[1]
                        predicts = net(x_data)
                        loss = paddle.nn.functional.cross_entropy(predicts, y_data)
                        # calc loss
                        acc = paddle.metric.accuracy(predicts, y_data, k=2)
                        avg_loss = paddle.mean(loss)
                        avg_acc = paddle.mean(acc)
                        avg_loss.backward()
                        if batch_id % 10 == 0:
                            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, avg_loss.numpy(), avg_acc.numpy()))
                        adam.step()
                        adam.clear_grad()
            train()

            # Example2: set Regularizer in parameters
            # Set L2 regularization in parameters.
            # Global regularizer does not take effect on my_conv2d for this case.
            from paddle.nn import Conv2d
            from paddle import ParamAttr
            from paddle.regularizer import L2Decay

            my_conv2d = Conv2d(
                    in_channels=10,
                    out_channels=10,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.0)),
                    bias_attr=False)
    """

    def __init__(self, coeff=0.0):
        super(L2Decay, self).__init__(coeff)
