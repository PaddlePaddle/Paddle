#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import warnings
import paddle
import os
import numpy as np
from paddle.fluid.framework import dygraph_only, _global_flags
from .base.distributed_strategy import DistributedStrategy
from .meta_optimizers import HybridParallelOptimizer, HeterParallelOptimizer
from paddle.fluid import core
from . import fleet


class Optimizer(object):

    def __init__(self):
        self.user_defined_optimizer = None

    def distributed_optimizer(self, optimizer, strategy=None):
        """
        Optimizer for distributed training.

        For the distributed training, this method would rebuild a new instance of DistributedOptimizer.
        Which has basic Optimizer function and special features for distributed training.

        Args:
            optimizer(Optimizer): The executor to run for init server.
            strategy(DistributedStrategy): Extra properties for distributed optimizer. 
                It is recommended to use DistributedStrategy in fleet.init(). The strategy
                here is for compatibility. If the strategy in fleet.distributed_optimizer() 
                is not None, then it will overwrite the DistributedStrategy in fleet.init(), 
                which will take effect in distributed training.

        Returns:
            Optimizer: Hybrid Optimizer or user define optimizer.

        Examples:

            .. code-block:: python

                import paddle
                import paddle.distributed.fleet as fleet
                fleet.init(is_collective=True)
                strategy = fleet.DistributedStrategy()
                optimizer = paddle.optimizer.SGD(learning_rate=0.001)
                optimizer = fleet.distributed_optimizer(optimizer, strategy=strategy)

        """
        self.user_defined_optimizer = optimizer

        if strategy is not None:
            if self.fleet._is_collective:
                warnings.warn(
                    "It is recommended to use DistributedStrategy "
                    "in fleet.init(). The strategy here is only for compatibility. "
                    "If the strategy in fleet.distributed_optimizer() is "
                    "not None, then it will overwrite the DistributedStrategy in fleet.init(), "
                    "which will take effect in distributed training.")
            self.fleet._user_defined_strategy = copy.deepcopy(strategy)

        self.fleet._context = {}

        if self.fleet.worker_num() > 1:
            if self.fleet._user_defined_strategy.heter_ccl_mode == False:
                return HybridParallelOptimizer(
                    optimizer, self.fleet._hcg,
                    self.fleet._user_defined_strategy)
            else:
                return HeterParallelOptimizer(optimizer,
                                              self.fleet._user_defined_strategy)
        else:
            return optimizer

    @dygraph_only
    def state_dict(self):
        """
        Get state dict information from optimizer.
        (Only work in dygraph mode)

        Returns: 
            state_dict(dict) : dict contains all the Tensor used by optimizer

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)
                state_dict = adam.state_dict()
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.state_dict()

    @dygraph_only
    def set_state_dict(self, state_dict):
        """
        Load optimizer state dict.
        (Only work in dygraph mode)

        Args: 
            state_dict(dict) : Dict contains all the Tensor needed by optimizer

        Returns:
            None

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)
                state_dict = adam.state_dict()
                paddle.save(state_dict, "paddle_dy")
                para_state_dict = paddle.load("paddle_dy")
                adam.set_state_dict(para_state_dict)
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.set_state_dict(state_dict)

    @dygraph_only
    def set_lr(self, value):
        """
        Set the value of the learning rate manually in the optimizer. 
        (Only work in dygraph mode)

        Args:
            value (float|Tensor): the value of learning rate

        Returns: 
            None 

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                lr_list = [0.2, 0.3, 0.4, 0.5, 0.6]
                for i in range(5):
                    adam.set_lr(lr_list[i])
                    lr = adam.get_lr()
                    print("current lr is {}".format(lr))
                # Print:
                #    current lr is 0.2
                #    current lr is 0.3
                #    current lr is 0.4
                #    current lr is 0.5
                #    current lr is 0.6
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.set_lr(value)

    @dygraph_only
    def get_lr(self):
        """
        Get current step learning rate.
        (Only work in dygraph mode)

        Returns:
            float: The learning rate of the current step.

        Examples:

            .. code-block:: python

                import numpy as np
                import paddle
                from paddle.distributed import fleet

                fleet.init(is_collective=True)

                value = np.arange(26).reshape(2, 13).astype("float32")
                a = paddle.to_tensor(value)

                layer = paddle.nn.Linear(13, 5)
                adam = paddle.optimizer.Adam(learning_rate=0.01, parameters=layer.parameters())

                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                lr = adam.get_lr()
                print(lr) # 0.01
        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.get_lr()

    @dygraph_only
    def step(self):
        """
        Execute the optimizer once.
        (Only work in dygraph mode)

        Returns:
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.nn as nn
                from paddle.distributed import fleet

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear1 = nn.Linear(10, 10)
                        self._linear2 = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear2(self._linear1(x))

                # 1. initialize fleet environment
                fleet.init(is_collective=True)

                # 2. create layer & optimizer
                layer = LinearNet()
                loss_fn = nn.MSELoss()
                adam = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=layer.parameters())

                # 3. get data_parallel model using fleet
                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()


        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.step()

    @dygraph_only
    def clear_grad(self):
        """
        Clear the gradients of all optimized parameters for model.
        (Only work in dygraph mode)

        Returns: 
            None

        Examples:

            .. code-block:: python

                import paddle
                import paddle.nn as nn
                from paddle.distributed import fleet

                class LinearNet(nn.Layer):
                    def __init__(self):
                        super(LinearNet, self).__init__()
                        self._linear1 = nn.Linear(10, 10)
                        self._linear2 = nn.Linear(10, 1)

                    def forward(self, x):
                        return self._linear2(self._linear1(x))

                # 1. initialize fleet environment
                fleet.init(is_collective=True)

                # 2. create layer & optimizer
                layer = LinearNet()
                loss_fn = nn.MSELoss()
                adam = paddle.optimizer.Adam(
                    learning_rate=0.001, parameters=layer.parameters())

                # 3. get data_parallel model using fleet
                adam = fleet.distributed_optimizer(adam)
                dp_layer = fleet.distributed_model(layer)

                # 4. run layer
                inputs = paddle.randn([10, 10], 'float32')
                outputs = dp_layer(inputs)
                labels = paddle.randn([10, 1], 'float32')
                loss = loss_fn(outputs, labels)

                print("loss:", loss.numpy())

                loss.backward()

                adam.step()
                adam.clear_grad()

        """
        # imitate target optimizer retrieval
        return self.user_defined_optimizer.clear_grad()


optimizer = Optimizer()
fleet.step = optimizer.step
fleet.clear_grad = optimizer.clear_grad
fleet.set_lr = optimizer.set_lr
fleet.get_lr = optimizer.get_lr
fleet.state_dict = optimizer.state_dict
fleet.set_state_dict = optimizer.set_state_dict


def opt_func(*args, **kwargs):
    if paddle.fluid.framework._non_static_mode():
        return optimizer.distributed_optimizer(*args, **kwargs)
    else:
        return fleet.distributed_optimizer(*args, **kwargs)


fleet.distributed_optimizer = opt_func
