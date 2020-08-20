# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
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
import numpy
import warnings
from .core import VarBase

__all__ = [
    'NoamLR', 'PiecewiseLR', 'NaturalExpLR', 'InverseTimeLR', 'PolynomialLR',
    'LinearLrWarmup', 'ExponentialLR', 'MultiStepLR', 'StepLR', 'LambdaLR',
    'ReduceLROnPlateau', 'CosineAnnealingLR'
]


class _LRScheduler(object):
    """
    """

    def __init__(self, learning_rate=0.1, last_epoch=-1, verbose=False):
        ""
        ""
        self.base_lr = learning_rate
        self.last_lr = learning_rate
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._var_name = None

        self.step()

    def __call__(self):
        """ 
        Return last computed learning rate on current epoch.
        """
        return self.last_lr

    def step(self, epoch=None):
        """
        compueted learning_rate and update it when invoked.
        """
        if epoch is None:
            self.last_epoch += 1
            self.last_lr = self.get_lr()
        else:
            self.last_epoch = epoch
            if hasattr(self, "_get_closed_form_lr"):
                self.last_lr = self._get_closed_form_lr()
            else:
                self.last_lr = self.get_lr()

        if self.verbose:
            print('Epoch {}: {} set learning rate to {}.'.format(
                self.last_epoch, self.__class__.__name__, self.last_lr))

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
                assert value.shape == [
                    1
                ], "shape of Variable in state_dict must be [1] {}".format(
                    value.shape)
                value = value.numpy()[0]
            state_dict[key] = value

        return state_dict

    # For those subclass who overload _LRScheduler, "last_epoch, last_lr" will be saved by default.
    # you can change it for your subclass.
    def _state_keys(self):
        """
        set the keys in self.__dict__ that are needed to be saved.
        """
        self.keys = ['last_epoch', 'last_lr']

    def set_dict(self, state_dict):
        """
        Loads the schedulers state.
        """
        self._state_keys()
        for key in self.keys:
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
            else:
                raise RuntimeError(
                    "Please check whether state_dict is correct for optimizer. Can't find [ {} ] in state_dict".
                    format(key))
        if len(state_dict) > len(self.keys):
            warnings.warn(
                "There are some unused values in state_dict. Maybe the optimizer have different 'LearningRateDecay' when invoking state_dict and set_dict"
            )

    def get_lr(self):
        # calculate by python float
        raise NotImplementedError


class NoamLR(_LRScheduler):
    """

    """

    def __init__(self,
                 d_model,
                 warmup_steps,
                 learning_rate=1.0,
                 last_epoch=-1,
                 verbose=False):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super(NoamLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            a = 1
        else:
            a = self.last_epoch**-0.5
        b = self.warmup_steps**-1.5 * self.last_epoch
        return self.base_lr * (self.d_model**-0.5) * min(a, b)


class PiecewiseLR(_LRScheduler):
    """
    """

    def __init__(self, boundaries, values, last_epoch=-1, verbose=False):
        self.boundaries = boundaries
        self.values = values
        super(PiecewiseLR, self).__init__(
            last_epoch=last_epoch, verbose=verbose)

    def get_lr(self):

        for i in range(len(self.boundaries)):
            if self.last_epoch < self.boundaries[i]:
                return self.values[i]
        return self.values[len(self.values) - 1]


class NaturalExpLR(_LRScheduler):
    """

    """

    def __init__(self, learning_rate, gama, last_epoch=-1, verbose=False):
        self.gama = gama
        super(NaturalExpLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * math.exp(-1 * self.gama * self.last_epoch)


class InverseTimeLR(_LRScheduler):
    """


    """

    def __init__(self, learning_rate, gama, last_epoch=-1, verbose=False):
        self.gama = gama
        super(InverseTimeLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr / (1 + self.gama * self.last_epoch)


class PolynomialLR(_LRScheduler):
    """
 
    """

    def __init__(self,
                 learning_rate,
                 decay_steps,
                 end_lr=0.0001,
                 power=1.0,
                 cycle=False,
                 last_epoch=-1,
                 verbose=False):
        self.decay_steps = decay_steps
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        super(PolynomialLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        tmp_epoch_num = self.last_epoch
        tmp_decay_steps = self.decay_steps
        if self.cycle:
            div_res = math.ceil(
                float(self.last_epoch) / float(self.decay_steps))

            if self.last_epoch == 0:
                div_res = 1
            tmp_decay_steps = self.decay_steps * div_res
        else:
            tmp_epoch_num = min(self.last_epoch, self.decay_steps)

        return (self.base_lr - self.end_lr) * (
            (1 - float(tmp_epoch_num) / float(tmp_decay_steps)
             )**self.power) + self.end_lr


class LinearLrWarmup(_LRScheduler):
    """
       
    """

    def __init__(self,
                 learning_rate,
                 warmup_steps,
                 start_lr,
                 end_lr,
                 last_epoch=-1,
                 verbose=False):
        type_check = isinstance(learning_rate, float) or isinstance(
            learning_rate, int) or isinstance(learning_rate, _LRScheduler)
        if not type_check:
            raise TypeError(
                "the type of learning_rate should be [int, float or _LRScheduler], the current type is {}".
                format(learning_rate))
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self.end_lr = end_lr
        assert end_lr > start_lr, "end_lr {} must be greater than start_lr {}".format(
            end_lr, start_lr)
        super(LinearLrWarmup, self).__init__(start_lr, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return (self.end_lr - self.start_lr) * float(
                self.last_epoch) / float(self.warmup_steps) + self.start_lr
        else:

            if isinstance(self.learning_rate, _LRScheduler):
                self.learning_rate.step()
                return self.learning_rate()

            return self.learning_rate


class ExponentialLR(_LRScheduler):
    """

    """

    def __init__(self, learning_rate, gama, last_epoch=-1, verbose=False):
        self.gama = gama
        super(ExponentialLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * (self.gama**self.last_epoch)


class MultiStepLR(_LRScheduler):
    """
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 gama=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(milestones, (tuple, list)):
            raise TypeError(
                "The type of 'milestones' in 'MultiStepDecay' must be 'tuple, list', but received %s."
                % type(milestones))

        if not all([
                milestones[i] < milestones[i + 1]
                for i in range(len(milestones) - 1)
        ]):
            raise ValueError('The elements of milestones must be incremented')
        if gama >= 1.0:
            raise ValueError('gama should be < 1.0.')

        self.milestones = milestones
        self.gama = gama
        super(MultiStepLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        for i in range(len(self.milestones)):
            if self.last_epoch < self.milestones[i]:
                return self.base_lr * (self.gama**i)
        return self.base_lr * (self.gama**len(self.milestones))


class StepLR(_LRScheduler):
    """

    """

    def __init__(self,
                 learning_rate,
                 step_size,
                 gama=0.1,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(step_size, int):
            raise TypeError(
                "The type of 'step_size' must be 'int', but received %s." %
                type(step_size))
        if gama >= 1.0:
            raise ValueError('gama should be < 1.0.')

        self.step_size = step_size
        self.gama = gama
        super(StepLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        i = self.last_epoch // self.step_size
        return self.base_lr * (self.gama**i)


class LambdaLR(_LRScheduler):
    """
 
    Sets the learning rate of ``optimizer`` to the initial lr times a multiplicative factor, and this multiplicative
    factor is computed by function ``lr_lambda`` . ``lr_lambda`` is funciton which receives ``epoch`` .

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
            with fluid.dygraph.guard():
                x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
                linear = fluid.dygraph.Linear(10, 10)
                input = fluid.dygraph.to_variable(x)
                scheduler = fluid.dygraph.LambdaDecay(0.5, lr_lambda=lambda x: 0.95**x)
                adam = fluid.optimizer.Adam(learning_rate = scheduler, parameter_list = linear.parameters())

                for epoch in range(6):
                    for batch_id in range(5):
                        out = linear(input)
                        loss = fluid.layers.reduce_mean(out)
                        adam.minimize(loss)
                    scheduler.epoch()

                    print("epoch:%d, current lr is %f" .format(epoch, adam.current_step_lr()))
                    # epoch:0, current lr is 0.5
                    # epoch:1, current lr is 0.475
                    # epoch:2, current lr is 0.45125

    """

    def __init__(self, learning_rate, lr_lambda, last_epoch=-1, verbose=False):
        if not callable(lr_lambda):
            raise TypeError(
                "The type of 'lr_lambda' in 'LambdaLR' must be 'function', but received %s."
                % type(lr_lambda))

        self.lr_lambda = lr_lambda
        super(LambdaLR, self).__init__(learning_rate, last_epoch, verbose)

    def get_lr(self):
        return self.base_lr * self.lr_lambda(self.last_epoch)


class ReduceLROnPlateau(_LRScheduler):
    """
    :api_attr: imperative

    Reduce learning rate when ``loss`` has stopped descending. Models often benefit from reducing the learning rate 
    by 2 to 10 times once model performance has no longer improvement.

    The ``loss`` is the one which has been pass into ``step`` , it must be 1-D Tensor with shape [1]. When ``loss`` 
    stop descending for a ``patience`` number of epochs, the learning rate will be reduced to ``learning_rate * decay_rate`` . 
    (Specially, ``mode`` can also be set to ``'max`` , in this case, when ``loss`` stop ascending for a ``patience`` number 
    of epochs, the learning rate will be reduced.)

    In addition, After each reduction, it will wait a ``cooldown`` number of epochs before resuming normal operation.

    Args:
        learning_rate (Variable|float|int): The initial learning rate. It can be set to python float or int number.
            If the type is Variable, it should be 1-D Tensor with shape [1], the data type can be 'float32' or 'float64'.
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the 
            learning rate will reduce when ``loss`` stops descending. Specially, if it's set to ``'max'`` ,  the learning 
            rate will reduce when ``loss`` stops ascending. Default: ``'min'`` .
        decay_rate (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * decay_rate`` . 
            It should be less than 1.0. Default: 0.1.
        patience (int, optional): When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced. 
            Default: 10.
        verbose (bool, optional): If ``True``, prints a message to stdout for each update. Default: ``False``.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` . 
            This make tiny changes of ``loss`` will be ignored. Default: 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss``
            is ``last_loss * threshold`` , where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum 
            change of ``loss`` is ``threshold`` . Default: ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Default: 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Default: 0.
        eps (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        dtype (str, optional): The data type used to create the learning rate variable. The data type can be set as
            'float32', 'float64'. Default: 'float32'. 
    
    Returns:
        Reduced learning rate.

    Examples:
    
    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np

        with fluid.dygraph.guard():
            x = np.random.uniform(-1, 1, [10, 10]).astype("float32")
            linear = fluid.dygraph.Linear(10, 10)
            input = fluid.dygraph.to_variable(x)

            reduce_lr = fluid.dygraph.ReduceLROnPlateau(
                                    learning_rate = 1.0,
                                    decay_rate = 0.5,
                                    patience = 5,
                                    verbose = True, 
                                    cooldown = 3)
            adam = fluid.optimizer.Adam(
                learning_rate = reduce_lr,
                parameter_list = linear.parameters())

            for epoch in range(10):
                total_loss = 0
                for bath_id in range(5):
                    out = linear(input)
                    loss = fluid.layers.reduce_mean(out)
                    total_loss += loss
                    adam.minimize(loss)
                
                avg_loss = total_loss/5

                # adjust learning rate according to avg_loss
                reduce_lr.step(avg_loss)
                lr = adam.current_step_lr()
                print("current avg_loss is %s, current lr is %s" % (avg_loss.numpy()[0], lr))

    """

    def __init__(self,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 verbose=False):
        mode = mode.lower()
        if mode not in ['min', 'max']:
            raise ValueError('mode ' + mode + ' is unknown!')
        self.mode = mode

        if factor >= 1.0:
            raise ValueError(
                'new_lr = origin_lr * decay_rate and decay_rate should be < 1.0.'
            )
        self.factor = factor

        threshold_mode = threshold_mode.lower()
        if threshold_mode not in ['rel', 'abs']:
            raise ValueError('threshold mode ' + threshold_mode +
                             ' is unknown!')
        self.threshold_mode = threshold_mode
        if not isinstance(learning_rate, float):
            raise TypeError(
                "The type of 'lr' in 'ReduceLROnPlateau' must be 'float', but received %s."
                % type(learning_rate))

        self.verbose = verbose
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.epsilon = epsilon

        self.cooldown_counter = 0
        self.best = None
        self.num_bad_epochs = 0

        # Can not call Parent __init__
        self.base_lr = learning_rate
        self.last_lr = learning_rate
        self.last_epoch = 0
        self.verbose = verbose
        self._var_name = None

    # "cooldown_counter / best_loss / num_bad_epochs / last_epoch / last_lr" will be stored.
    def _state_keys(self):
        self.keys = [
            'cooldown_counter', 'best', 'num_bad_epochs', 'last_epoch',
            'last_lr'
        ]

    def step(self, metrics, epoch=None):
        """
        It should be invoked on each epoch. Update the learning rate in optimizer according to ``loss`` .  
        The new learning rate will take effect on next call to ``optimizer.minimize`` .

        Args:
            loss (Variable): A ``Variable`` that will be monitored to determine whether the learning rate will reduce. 
                If it stop descending for a ``patience`` number of epochs, the learning rate will reduce. It should 
                be 1-D Tensor with shape [1]. 
                Specially, if ``mode`` has been set to ``'max'`` ,  the learning rate will reduce when it stops ascending.
        Returns:
            None
        
        Examples:
            Please refer to the example of current LearningRateDecay.
        """
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = epoch

        # loss must be 1-D Tensor with shape [1]
        if isinstance(metrics, (VarBase, numpy.ndarray)):
            assert len(metrics.shape) == 1 and metrics.shape[0] == 1, "the metrics.shape " \
                "should be (1L,), but the current metrics.shape is {}. Maybe that "  \
                "you should call paddle.mean to process it first.".format(loss.shape)
        elif not isinstance(metrics,
                            (int, float, numpy.float32, numpy.float64)):
            raise TypeError(
                "metrics must be 'int', 'float', 'np.float', 'numpy.ndarray' or 'paddle.Tensor', but receive {}".
                format(type(metrics)))

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        else:
            if self.best is None or self._is_better(metrics, self.best):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs > self.patience:
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
                new_lr = max(self.last_lr * self.factor, self.min_lr)
                if self.last_lr - new_lr > self.epsilon:
                    self.last_lr = new_lr
                    if self.verbose:
                        print('Epoch {}: {} adjusting learning rate to {}.'.
                              format(self.last_epoch, self.__class__.__name__,
                                     self.last_lr))

    def _is_better(self, current, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            return current < best - best * self.threshold

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return current < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            return current > best + best * self.threshold

        else:
            return current > best + self.threshold


class CosineAnnealingLR(_LRScheduler):
    """
    ...
    """

    def __init__(self,
                 learning_rate,
                 T_max,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        if not isinstance(T_max, int):
            raise TypeError(
                "The type of 'T_max' in 'CosineAnnealingLR' must be 'int', but received %s."
                % type(T_max))
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingLR' must be 'float, int', but received %s."
                % type(eta_min))
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super(CosineAnnealingLR, self).__init__(learning_rate, last_epoch,
                                                verbose)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(
                math.pi / self.T_max)) / 2

        return (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / (
            1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) * (
                self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max)) / 2
