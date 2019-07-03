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

from __future__ import print_function

import math

from .. import unique_name

__all__ = [
    'NoamDecay', 'PiecewiseDecay', 'NaturalExpDecay', 'ExponentialDecay',
    'InverseTimeDecay', 'PolynomialDecay', 'CosineDecay'
]


class LearningRateDecay(object):
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
        lr = layers.create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(lr),
            dtype=self.dtype,
            persistable=False)
        return lr

    def step(self):
        raise NotImplementedError()


class PiecewiseDecay(LearningRateDecay):
    """
    piecewise decay scheduler

    The algorithm can be described as the code below.

    .. code-block:: text

      boundaries = [10000, 20000]
      values = [1.0, 0.5, 0.1]
      if step < 10000:
          learning_rate = 1.0
      elif 10000 <= step < 20000:
          learning_rate = 0.5
      else:
          learning_rate = 0.1
    Args:
        boundaries: A list of steps numbers.
        values: A list of learning rate values that will be picked during
            different step boundaries.
        begin: The begin step to initilize the self.step_num
        step: The step_size using when calculate the new step_num (Defalult is 1)
        dtype: The dtype used to create the learning rate variable

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          with fluid.dygraph.guard():
              optimizer = fluid.optimizer.SGD(
                 learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0) )
    """

    def __init__(self, boundaries, values, begin, step=1, dtype='float32'):
        super(PiecewiseDecay, self).__init__(begin, step, dtype)
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


class NaturalExpDecay(LearningRateDecay):
    """
    Applies natural exponential decay to the initial learning rate.
    
    .. code-block:: python

        if not staircase:
            decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))
        else:
            decayed_learning_rate = learning_rate * exp(- decay_rate * (global_step / decay_steps))

    Args:
        learning_rate: A scalar float32 value or a Variable. This
          will be the initial learning rate during training
        decay_steps: A Python `int32` number.
        decay_rate: A Python `float` number.
        staircase: Boolean. If set true, decay the learning rate every decay_steps.
        begin: A Python 'int32' number, the begin step (Default is 0)
        step: A Python 'int32' number, the step size (Default is 1)
        dtype: A Python 'str', the dtype used to create learning rate variable (Default is 'float32')

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              sgd_optimizer = fluid.optimizer.SGD(
        	      learning_rate=fluid.dygraph.NaturalExpDecay(
	    	            learning_rate=base_lr,
        		    decay_steps=10000,
		            decay_rate=0.5,
		            staircase=True))

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(NaturalExpDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)
        decayed_lr = self.learning_rate * layers.exp(-1 * self.decay_rate *
                                                     div_res)

        return decayed_lr


class ExponentialDecay(LearningRateDecay):
    """
    Applies exponential decay to the learning rate.

    When training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, the learning rate will be decayed by
    'decay_rate' every 'decay_steps' steps.
    
    .. code-block:: python

        if staircase == True:
            decayed_learning_rate = learning_rate * decay_rate ^ floor(global_step / decay_steps)
        else:
            decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)

    Args:
        learning_rate(Variable|float): The initial learning rate.
        decay_steps(int): See the decay computation above.
        decay_rate(float): The decay rate. See the decay computation above.
        staircase(Boolean): If True, decay the learning rate at discrete intervals.
                            Default: False
        begin(int): The begin step (default is 0)
        step(int): The step size (default is 1)
        dtype(str): The dtype used to create learning rate (default is 'float32')

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              sgd_optimizer = fluid.optimizer.SGD(
    	            learning_rate=fluid.dygraph.ExponentialDecay(
		        learning_rate=base_lr,
    		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True))

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(ExponentialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate * (self.decay_rate**div_res)

        return decayed_lr


class InverseTimeDecay(LearningRateDecay):
    """
    Applies inverse time decay to the initial learning rate.

    When training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, an inverse decay function will be
    applied to the initial learning rate.

    >>> if staircase == True:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * floor(global_step / decay_step))
    >>> else:
    >>>     decayed_learning_rate = learning_rate / (1 + decay_rate * global_step / decay_step)

    Args:
        learning_rate(Variable|float): The initial learning rate.
        decay_steps(int): See the decay computation above.
        decay_rate(float): The decay rate. See the decay computation above.
        staircase(Boolean): If True, decay the learning rate at discrete intervals.
                            Default: False
        begin(int): The begin step (default is 0)
        step(int): The step size (default is 1)
        dtype(str): The dtype used to create learning rate (default is 'float32')

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              sgd_optimizer = fluid.optimizer.SGD(
	          learning_rate=fluid.dygraph.InverseTimeDecay(
		        learning_rate=base_lr,
		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True))

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 decay_rate,
                 staircase=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(InverseTimeDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase

    def step(self):
        from .. import layers
        div_res = self.create_lr_var(self.step_num / self.decay_steps)
        if self.staircase:
            div_res = layers.floor(div_res)

        decayed_lr = self.learning_rate / (1 + self.decay_rate * div_res)

        return decayed_lr


class PolynomialDecay(LearningRateDecay):
    """
    Applies polynomial decay to the initial learning rate.

    .. code-block:: text

     if cycle:
       decay_steps = decay_steps * ceil(global_step / decay_steps)
     else:
       global_step = min(global_step, decay_steps)
       decayed_learning_rate = (learning_rate - end_learning_rate) *
            (1 - global_step / decay_steps) ^ power + end_learning_rate

    Args:
        learning_rate(Variable|float32): A scalar float32 value or a Variable. This
          will be the initial learning rate during training.
        decay_steps(int32): A Python `int32` number.
        end_learning_rate(float): A Python `float` number.
        power(float): A Python `float` number.
        cycle(bool): If set true, decay the learning rate every decay_steps.
        begin(int): The begin step (default is 0)
        step(int): The step size (default is 1)
        dtype(str): The dtype used to create learning rate (default is 'float32')

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          start_lr = 0.01
          total_step = 5000
          end_lr = 0
          with fluid.dygraph.guard():
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.PolynomialDecay(
                  start_lr, total_step, end_lr, power=1.0) )

    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_learning_rate=0.0001,
                 power=1.0,
                 cycle=False,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(PolynomialDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.cycle = cycle

    def step(self):
        from .. import layers
        tmp_step_num = self.step_num
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = layers.ceil(
                self.create_lr_var(tmp_step_num / float(self.decay_steps)))

            if tmp_step_num == 0:
                div_res = self.create_lr_var(1.0)
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_step_num = self.create_lr_var(tmp_step_num
                                              if tmp_step_num < self.decay_steps
                                              else self.decay_steps)

        decayed_lr = (self.learning_rate - self.end_learning_rate) * \
            ((1 - tmp_step_num / tmp_decay_steps) ** self.power) + self.end_learning_rate
        return decayed_lr


class CosineDecay(LearningRateDecay):
    """
    Applies cosine decay to the learning rate.

    when training a model, it is often recommended to lower the learning rate as the
    training progresses. By using this function, the learning rate will be decayed by
    following cosine decay strategy.

    .. math::

	decayed\_lr = learning\_rate * 0.5 * (math.cos * (epoch * \\frac{math.pi}{epochs} ) + 1)
    
    Args:
        learning_rate(Variable|float): The initial learning rate.
        step_each_epoch(int): the number of steps in an epoch.
        epochs(int): the number of epochs.
        begin(int): The begin step (default is 0).
        step(int): The step size (default is 1).
        dtype(str): The dtype used to create learning rate (default is 'float32').

    Examples:
	.. code-block:: python

  	    base_lr = 0.1
            with fluid.dygraph.guard():
                optimizer  = fluid.optimizer.SGD(
        	    learning_rate = fluid.dygraph.CosineDecay(
	                    base_lr, 10000, 120) )
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 begin=0,
                 step=1,
                 dtype='float32'):
        super(CosineDecay, self).__init__(begin, step, dtype)
        self.learning_rate = learning_rate
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs

    def step(self):
        from .. import layers
        cur_epoch = layers.floor(
            self.create_lr_var(self.step_num / self.step_each_epoch))
        decayed_lr = self.learning_rate * 0.5 * (
            layers.cos(cur_epoch * math.pi / self.epochs) + 1)
        return decayed_lr


class NoamDecay(LearningRateDecay):
    """
    Noam decay method. The numpy implementation of noam decay as follows.

    .. code-block:: python
      
      import numpy as np
      # set hyper parameters
      d_model = 2
      current_steps = 20
      warmup_steps = 200
      # compute
      lr_value = np.power(d_model, -0.5) * np.min([
                              np.power(current_steps, -0.5),
                              np.power(warmup_steps, -1.5) * current_steps])

    Please reference `attention is all you need
    <https://arxiv.org/pdf/1706.03762.pdf>`_.

    Args:
        d_model(Variable): The dimensionality of input and output of model.

        warmup_steps(Variable): A super parameter.
        begin(int): The begin step (default is 0)
        step(int): The step size (default is 1)
        dtype(str): The dtype used to create learning rate (default is 'float32')

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          warmup_steps = 100
          learning_rate = 0.01
          with fluid.dygraph.guard():
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.NoamDecay(
                         1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps) )
    """

    def __init__(self, d_model, warmup_steps, begin=1, step=1, dtype='float32'):
        super(NoamDecay, self).__init__(begin, step, dtype)
        self.d_model = d_model
        self.warmup_steps = warmup_steps

    def step(self):
        from .. import layers
        a = self.create_lr_var(self.step_num**-0.5)
        b = self.create_lr_var((self.warmup_steps**-1.5) * self.step_num)
        lr_value = (self.d_model**-0.5) * layers.elementwise_min(a, b)
        return lr_value
