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
          boundaries = [10000, 20000]
          values = [1.0, 0.5, 0.1]
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding( [10, 10] )
              optimizer = fluid.optimizer.SGD(
                 learning_rate=fluid.dygraph.PiecewiseDecay(boundaries, values, 0),
                 parameter_list = emb.parameters() )
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
    
    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * e^{y} 

    If staircase is set to False, then:

    .. math::

        y = - decay\_rate * \\frac{global\_step}{decay\_steps}

    If staircase is set to True, then:

    .. math::

        y = - decay\_rate * math.floor(\\frac{global\_step}{decay\_steps}) 

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(int): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

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

    The algorithm can be described as following.
    
    .. math::

        decayed\_learning\_rate = learning\_rate * decay\_rate ^ y 

    If staircase is set to False, then:

    .. math::

        y = \\frac{global\_step}{decay\_steps} 

    If staircase is set to True, then:

    .. math::

        y = math.floor(\\frac{global\_step}{decay\_steps})


    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(float): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

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

    The algorithm can be described as following.
    If staircase is set to False, then:

    .. math::

        decayed\_learning\_rate = \\frac{learning\_rate}{1 + decay\_rate * \\frac{global\_step}{decay\_step}}  

    If staircase is set to True, then:

    .. math::

        decayed\_learning\_rate = \\frac{learning\_rate}{1 + decay\_rate * math.floor(\\frac{global\_step}{decay\_step})}

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int): The decay step size. It determines the decay cycle.
        decay_rate(float): The decay rate.
        staircase(bool, optional): If set to True, decay the learning rate at discrete intervals. The 
            default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be 
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          base_lr = 0.1
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding([10, 10])
              sgd_optimizer = fluid.optimizer.SGD(
	          learning_rate=fluid.dygraph.InverseTimeDecay(
		        learning_rate=base_lr,
		        decay_steps=10000,
		        decay_rate=0.5,
		        staircase=True),
                  parameter_list = emb.parameters())

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

    The algorithm can be described as following.

    If cycle is set to True, then:

    .. math::

        decay\_steps & = decay\_steps * math.ceil(\\frac{global\_step}{decay\_steps}) 

        decayed\_learning\_rate & = (learning\_rate-end\_learning\_rate)*(1-\\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

    If cycle is set to False, then:

    .. math::

        global\_step & = min(global\_step, decay\_steps) 

        decayed\_learning\_rate & = (learning\_rate-end\_learning\_rate)*(1-\\frac{global\_step}{decay\_steps})^{power}+end\_learning\_rate

    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        decay_steps(int32): The decay step size. It determines the decay cycle.
        end_learning_rate(float, optional): The minimum final learning rate. The default value is 0.0001.
        power(float, optional): Power of polynomial. The default value is 1.0.
        cycle(bool, optional): If set true, decay the learning rate every decay_steps. The default value is False.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          start_lr = 0.01
          total_step = 5000
          end_lr = 0
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding( [10, 10])
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.PolynomialDecay(
                  start_lr, total_step, end_lr, power=1.0),
                  parameter_list = emb.parameters())

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

    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = learning\_rate * 0.5 * (math.cos(global\_step * \\frac{math.pi}{step\_each\_epoch} ) + 1)
    
    Parameters:
        learning_rate(Variable|float): The initial learning rate. If the type 
            is Variable, it's a tensor with shape [1], the data type can be  
            float32 or float64. It also can be set to python int number.
        step_each_epoch(int): The number of steps in an epoch.
        epochs(int): The number of epochs.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

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
    Applies Noam decay to the initial learning rate. 

    The algorithm can be described as following.

    .. math::

        decayed\_learning\_rate = d_{model}^{-0.5} * min(global\_step^{-0.5}, global\_step * warmup\_steps^{-1.5})

    Please reference `attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`_ 

    Parameters:
        d$_{model}$(Variable|int): The dimensionality of input and output feature vector of model. If type is Variable, 
            it's a tensor with shape [1] and the data type can be int32 or int64. The type can also be python int.
        warmup_steps(Variable|int): The number of warmup steps. A super parameter. If type is Variable, 
            it's a tensor with shape [1] and the data type can be int32 or int64. The type can also be python int.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.

    Returns:
        None.

    Examples:
        .. code-block:: python

          import paddle.fluid as fluid
          warmup_steps = 100
          learning_rate = 0.01
          with fluid.dygraph.guard():
              emb = fluid.dygraph.Embedding([10, 10])
              optimizer  = fluid.optimizer.SGD(
                  learning_rate = fluid.dygraph.NoamDecay(
                         1/(warmup_steps *(learning_rate ** 2)),
                         warmup_steps),
                  parameter_list = emb.parameters())
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


class LinearLrWarmup(LearningRateDecay):
    """
    This operator use the linear learning rate warm up strategy to adjust the learning rate preliminarily before the normal learning rate scheduling.
    For more information, please refer to `Bag of Tricks for Image Classification with Convolutional Neural Networks <https://arxiv.org/abs/1812.01187>`_
    
    When global_step < warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            linear_step = end_lr - start_lr
            lr = start_lr + linear_step * (global_step / warmup_steps)
    
    where start_lr is the initial learning rate, and end_lr is the final learning rate;
    
    When global_step >= warmup_steps, learning rate is updated as:
    
    .. code-block:: text
    
            lr = learning_rate
    
    where lr is the learning_rate after warm-up.
    
    Args:
        learning_rate (Variable|float): Learning_rate after warm-up, it could be 1D-Tensor or single value with the data type of float32.
        warmup_steps (int): Steps for warm up.
        start_lr (float): Initial learning rate of warm up.
        end_lr (float): Final learning rate of warm up.
        begin(int, optional): The begin step. The initial value of global_step described above. The default value is 0.
        step(int, optional): The step size used to calculate the new global_step in the description above.
            The default value is 1.
        dtype(str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. The default value is 'float32'.
    
    Returns:
        Variable: Warm-up learning rate with the same data type as learning_rate.
    
    
    Examples:
    
    .. code-block:: python
    
        import paddle.fluid as fluid
    
        learning_rate = 0.1 
        warmup_steps = 50
        start_lr = 1. / 3.
        end_lr = 0.1

        with fluid.dygraph.guard(): 
            lr_decay = fluid.dygraph.LinearLrWarmup( learning_rate, warmup_steps, start_lr, end_lr)
    
       
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 begin=1,
                 step=1,
                 dtype='float32'):
        super(LinearLrWarmup, self).__init__(begin, step, dtype)
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, LearningRateDecay)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or LearningRateDecay], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        self.lr_ratio_before_warmup = (
            float(end_lr) - float(start_lr)) / float(warmup_steps)

    def step(self):
        base_lr = self.learning_rate
        if isinstance(self.learning_rate, LearningRateDecay):
            base_lr = base_lr()

        from .. import layers
        if self.step_num < self.warmup_steps:
            return self.lr_ratio_before_warmup * self.step_num
        else:
            return base_lr
