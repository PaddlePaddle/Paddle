# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved
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

import math
import warnings
import numpy as np

import paddle
from .. import unique_name
from ..framework import Variable
from ..data_feeder import check_type

__all__ = [
    'PiecewiseDecay',
    'StepDecay',
    'MultiStepDecay',
    'LambdaDecay',
]


class LearningRateDecay:
    """
    Base class of learning rate decay

    Define the common interface of an LearningRateDecay.
    User should not use this class directly,
    but need to use one of it's implementation.
    """

    def __init__(self, begin=0, step=1, dtype='float32'):
        self.step_num = begin
        self.step_size = step
        self.dtype = dtype

    def __call__(self):
        lr = self.step()
        if isinstance(lr, float):
            lr = self.create_lr_var(lr)
        self.step_num += self.step_size
        return lr

    def create_lr_var(self, lr):
        """
        convert lr from float to variable

        Args:
            lr: learning rate
        Returns:
            learning rate variable
        """
        from .. import layers

        lr = paddle.static.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=False,
        )
        return lr

    # Note: If you want to change what optimizer.state_dict stores, just overwrite this functions,
    # "self.step_num" will be stored by default.
    def state_dict(self):
        """
        Returns the state of the scheduler as a :class:`dict`.

        It is a subset of self.__dict__ .
        """
        self._state_keys()
        state_dict = {}
        for key in self.keys:
            if key not in self.__dict__:
                continue
            value = self.__dict__[key]
            if isinstance(value, Variable):
                assert (
                    value.size == 1
                ), "the size of Variable in state_dict must be 1, but its size is {} with shape {}".format(
                    value.size, value.shape
                )
                value = value.item()
            state_dict[key] = value

        return state_dict

    def _state_keys(self):
        """
        set the keys in self.__dict__ that are needed to be saved.
        """
        self.keys = ['step_num']

    def set_state_dict(self, state_dict):
        """
        Loads the schedulers state.
        """
        self._state_keys()
        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                raise RuntimeError(
                    "Please check whether state_dict is correct for optimizer. Can't find [ {} ] in state_dict".format(
                        key
                    )
                )
        if len(state_dict) > len(self.keys):
            warnings.warn(
                "There are some unused values in state_dict. Maybe the optimizer have different 'LearningRateDecay' when invoking state_dict and set_dict"
            )

    # [aliases] Compatible with old method names
    set_dict = set_state_dict

    def step(self):
        raise NotImplementedError()


class PiecewiseDecay(LearningRateDecay):
    """
    :api_attr: imperative

    Piecewise decay scheduler.

    The algorithm can be described as the code below.

    .. code-block:: text

        boundaries = [10000, 20000]
        values = [1.0, 0.5, 0.1]
        if global_step < 10000:
            learning_rate = 1.0
        elif 10000 <= global_step < 20000:
            learning_rate = 0.5
        else:
            learning_rate = 0.1

    Parameters:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during
            different step boundaries. The type of element in the list is python float.
        begin(int): The begin step to initialize the global_step in the description above.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          import paddle
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          with fluid.dygraph.guard():
              emb = paddle.nn.Embedding(10, 10)
              optimizer = fluid.optimizer.SGD(
                 learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
                 parameter_list = emb.parameters() )
    """

    def __init__(self, boundaries, values, begin, step=1, dtype='float32'):
        super().__init__(begin, step, dtype)
        self.boundaries = boundaries
        self.values = values

        self.vars = []
        for value in values:
            self.vars.append(value)

    def step(self):
        for i in range(len(self.boundaries)):
            if self.step_num < self.boundaries[i]:
                return self.vars[i]
        return self.create_lr_var(self.vars[len(self.values) - 1])


class _LearningRateEpochDecay(LearningRateDecay):
    """
    :api_attr: imperative

    Base class of learning rate decay, which is updated each epoch.

    Define the common interface of an _LearningRateEpochDecay.
    User should not use this class directly,
    but need to use one of it's implementation. And invoke method: `epoch()` each epoch.
    """

    def __init__(self, learning_rate, dtype=None):
        if not isinstance(learning_rate, (float, int)):
            raise TypeError(
                "The type of 'learning_rate' must be 'float, int', but received %s."
                % type(learning_rate)
            )
        if learning_rate < 0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))

        self.base_lr = float(learning_rate)

        self.epoch_num = -1
        self.dtype = dtype
        if dtype is None:
            self.dtype = "float32"
        self.learning_rate = self.create_lr_var(self.base_lr)

        self.epoch()

    # For those subclass who overload _LearningRateEpochDecay, "self.epoch_num/learning_rate" will be stored by default.
    # you can change it for your subclass.
    def _state_keys(self):
        self.keys = ['epoch_num', 'learning_rate']

    def __call__(self):
        """
        Return last computed learning rate on current epoch.
        """
        if not isinstance(self.learning_rate, Variable):
            self.learning_rate = self.create_lr_var(self.learning_rate)
        return self.learning_rate

    def epoch(self, epoch=None):
        """
        compueted learning_rate and update it when invoked.
        """
        if epoch is None:
            self.epoch_num += 1
        else:
            self.epoch_num = epoch

        self.learning_rate = self.get_lr()

    def get_lr(self):
        raise NotImplementedError


class StepDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Decays the learning rate of ``optimizer`` by ``decay_rate`` every ``step_size`` number of epoch.

    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.5
        step_size = 30
        decay_rate = 0.1

        learning_rate = 0.5     if epoch < 30
        learning_rate = 0.05    if 30 <= epoch < 60
        learning_rate = 0.005   if 60 <= epoch < 90
        ...

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        step_size (int): Period of learning rate decay.
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` .
            It should be less than 1.0. Default: 0.1.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.StepDecay(0.5, step_size=3)
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(9):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = paddle.mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.5
                    # epoch:2, current lr is 0.5
                    # epoch:3, current lr is 0.05
                    # epoch:4, current lr is 0.05
                    # epoch:5, current lr is 0.05
                    # epoch:6, current lr is 0.005
                    # epoch:7, current lr is 0.005
                    # epoch:8, current lr is 0.005

    """

    def __init__(self, learning_rate, step_size, decay_rate=0.1):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s."
                % type(step_size)
            )
        if decay_rate >= 1.0:
            raise ValueError('decay_rate should be < 1.0.')

        self.step_size = step_size
        self.decay_rate = decay_rate
        super().__init__(learning_rate)

    def get_lr(self):
        decay_rate = self.create_lr_var(self.decay_rate)
        i = self.epoch_num // self.step_size
        return self.base_lr * (decay_rate**i)


class MultiStepDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Decays the learning rate of ``optimizer`` by ``decay_rate`` once ``epoch`` reaches one of the milestones.

    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.5
        milestones = [30, 50]
        decay_rate = 0.1
        if epoch < 30:
            learning_rate = 0.5
        elif epoch < 50:
            learning_rate = 0.05
        else:
            learning_rate = 0.005

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        milestones (tuple|list): List or tuple of each boundaries. Must be increasing.
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` .
            It should be less than 1.0. Default: 0.1.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.MultiStepDecay(0.5, milestones=[3, 5])
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = paddle.mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:{}, current lr is {}" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.5
                    # epoch:2, current lr is 0.5
                    # epoch:3, current lr is 0.05
                    # epoch:4, current lr is 0.05
                    # epoch:5, current lr is 0.005

    """

    def __init__(self, learning_rate, milestones, decay_rate=0.1):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones)
            )

        if not all(
            [
                milestones[i] < milestones[i + 1]
                for i in range(len(milestones) - 1)
            ]
        ):
            raise ValueError('The elements of milestones must be incremented')
        if decay_rate >= 1.0:
            raise ValueError('decay_rate should be < 1.0.')

        self.milestones = milestones
        self.decay_rate = decay_rate
        super().__init__(learning_rate)

    def get_lr(self):
        decay_rate = self.create_lr_var(self.decay_rate)
        for i in range(len(self.milestones)):
            if self.epoch_num < self.milestones[i]:
                return self.base_lr * (decay_rate**i)

        return self.base_lr * (decay_rate ** len(self.milestones))


class LambdaDecay(_LearningRateEpochDecay):
    """
    :api_attr: imperative

    Sets the learning rate of ``optimizer`` to the initial lr times a multiplicative factor, and this multiplicative
    factor is computed by function ``lr_lambda`` . ``lr_lambda`` is function which receives ``epoch`` .

    The algorithm can be described as the code below.

    .. code-block:: text

        learning_rate = 0.5        # init learning_rate
        lr_lambda = lambda epoch: 0.95 ** epoch

        learning_rate = 0.5        # epoch 0
        learning_rate = 0.475      # epoch 1
        learning_rate = 0.45125    # epoch 2

    Parameters:
        learning_rate (float|int): The initial learning rate. It can be set to python float or int number.
        lr_lambda (function): A function which computes a multiplicative factor given an integer parameter ``epoch`` , and
            then multiply the initial learning rate by this multiplicative factor.

    Returns:
        None.

    Examples:
        .. code-block:: python

            import paddle.fluid as fluid
            import numpy as np
            import paddle
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = paddle.nn.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.LambdaDecay(0.5, lr_lambda=lambda x: 0.95**x)
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = paddle.mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:%d, current lr is %f" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.475
                    # epoch:2, current lr is 0.45125

    """

    def __init__(self, learning_rate, lr_lambda):
        if not callable(lr_lambda):
            raise TypeError(
                "The type of 'lr_lambda' in 'LambdaDecay' must be 'function', but received %s."
                % type(lr_lambda)
            )

        self.lr_lambda = lr_lambda
        super().__init__(learning_rate)

    def get_lr(self):
        base_lr = self.create_lr_var(self.base_lr)

        return self.base_lr * self.lr_lambda(self.epoch_num)
