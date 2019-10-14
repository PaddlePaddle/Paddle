# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from ... import default_main_program
from ... import default_startup_program
from ... import layers
from ... import unique_name
from . import fp16_utils
from .fp16_utils import update_loss_scaling, rewrite_program
from .fp16_utils import update_role_var_grad
from .fp16_lists import AutoMixedPrecisionLists

__all__ = ["decorate"]


class OptimizerWithMixedPrecison(object):
    """
    Optimizer with mixed-precision (MP) training. This is a wrapper of a common 
    optimizer, plus the support of mixed-precision pretraining. The object
    of this class almost has the same behavior as the common optimizer, with the 
    methods `minimize()`, `backward()`, `apply_gradients()` implemented. 
    Additionally, it enables the MP training automatically, i.e, the creation 
    and maintenance of master parameters, scaling of loss, etc.

    Args:
        optimizer (Optimizer): A common Optimizer object.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        init_loss_scaling (float): The initial loss scaling factor.
        use_dynamic_loss_scaling (bool): Whether to use dynamic loss scaling.
        incr_every_n_steps(int): Increases loss scaling every n consecutive 
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n 
                                      accumulated steps with nan or 
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           the loss scaling.

    """

    def __init__(self, optimizer, amp_lists, init_loss_scaling,
                 use_dynamic_loss_scaling, incr_every_n_steps,
                 decr_every_n_nan_or_inf, incr_ratio, decr_ratio):
        self._optimizer = optimizer
        self._amp_lists = amp_lists
        self._param_grads = None
        self._train_program = default_main_program()
        self._startup_prog = default_startup_program()
        self._loss_scaling = layers.create_global_var(
            name=unique_name.generate("loss_scaling"),
            shape=[1],
            value=init_loss_scaling,
            dtype='float32',
            persistable=True)
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling
        if self._use_dynamic_loss_scaling:
            self._incr_every_n_steps = layers.fill_constant(
                shape=[1], dtype='int32', value=incr_every_n_steps)
            self._decr_every_n_nan_or_inf = layers.fill_constant(
                shape=[1], dtype='int32', value=decr_every_n_nan_or_inf)
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._num_good_steps = layers.create_global_var(
                name=unique_name.generate("num_good_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)
            self._num_bad_steps = layers.create_global_var(
                name=unique_name.generate("num_bad_steps"),
                shape=[1],
                value=0,
                dtype='int32',
                persistable=True)

        # Ensure the data type of learning rate vars is float32 (same as the 
        # master parameter dtype)
        if isinstance(optimizer._learning_rate, float):
            optimizer._learning_rate_map[default_main_program()] = \
                        layers.create_global_var(
                        name=unique_name.generate("learning_rate"),
                        shape=[1],
                        value=float(optimizer._learning_rate),
                        dtype='float32',
                        persistable=True)

    def get_loss_scaling(self):
        """Return the real-time loss scaling factor.
        """
        return self._loss_scaling

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        Backward propogation or auto differentiation for gradients' computation.

        Args:
            loss (Variable): The loss Variable to minimize.
            startup_program (Program|None): The startup Program for initializing 
                                       parameters in `parameter_list`.
            parameter_list (list|None): A list of Variables to update.
            no_grad_set (set|None): A set of Variables should be ignored.
            callbacks (list|None): A list of callables to run when appending 
                                   backward operator for one parameter.

        Returns:
            A list of (param, grad), which is a tuple of a parameter and its 
            gradient respectively, and the scaled loss.
        """
        rewrite_program(self._train_program, self._amp_lists)
        scaled_loss = loss * self._loss_scaling
        self._params_grads = self._optimizer.backward(
            scaled_loss, startup_program, parameter_list, no_grad_set,
            callbacks)
        update_role_var_grad(self._train_program, self._params_grads)
        scaled_params_grads = []
        for p, g in self._params_grads:
            with self._train_program._optimized_guard([p, g]):
                scaled_g = g / self._loss_scaling
                scaled_params_grads.append([p, scaled_g])

        return scaled_params_grads

    def apply_gradients(self, scaled_params_grads):
        """
        Check scaled gradients to determine whether to update loss scaling and update 
        parameters by their scaled gradients, 
  
        Args:
            scaled_params_grads (list): A list of params and scaled grads.
    
        Returns:
            A list of optimize operators.
        """

        if self._use_dynamic_loss_scaling:

            grads = [layers.reduce_sum(g) for [_, g] in scaled_params_grads]
            all_grads = layers.concat(grads)
            all_grads_sum = layers.reduce_sum(all_grads)
            is_overall_finite = layers.isfinite(all_grads_sum)

            update_loss_scaling(is_overall_finite, self._loss_scaling,
                                self._num_good_steps, self._num_bad_steps,
                                self._incr_every_n_steps,
                                self._decr_every_n_nan_or_inf, self._incr_ratio,
                                self._decr_ratio)

            # apply_gradient append all ops in global block, thus we shouldn't
            # apply gradient in the switch branch.
            with layers.Switch() as switch:
                with switch.case(is_overall_finite):
                    pass
                with switch.default():
                    for _, g in scaled_params_grads:
                        layers.assign(layers.zeros_like(g), g)

        optimize_ops = self._optimizer.apply_gradients(scaled_params_grads)

        return optimize_ops

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        """
        Perform optimization by minimizing the given loss.

        Args:
            loss (Variable): The loss Variable.
            startup_program (Program): startup_program for initializing parameters
                in `parameter_list`.
            parameter_list (list): list of Variables to update.
            no_grad_set (set|None): set of Variables should be ignored.

        Returns:
            The scaled loss by scaling factor, the list of optimize ops, and a
            list of scaled parameters and gradients.
        """
        scaled_params_grads = self.backward(
            loss,
            startup_program=startup_program,
            parameter_list=parameter_list,
            no_grad_set=no_grad_set)

        optimize_ops = self.apply_gradients(scaled_params_grads)

        return optimize_ops, scaled_params_grads


def decorate(optimizer,
             amp_lists=None,
             init_loss_scaling=1.0,
             incr_every_n_steps=1000,
             decr_every_n_nan_or_inf=2,
             incr_ratio=2.0,
             decr_ratio=0.8,
             use_dynamic_loss_scaling=True):
    """ 
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        amp_lists (AutoMixedPrecisionLists): An AutoMixedPrecisionLists object.
        init_loss_scaling(float): The initial loss scaling factor.
        incr_every_n_steps(int): Increases loss scaling every n consecutive 
                                 steps with finite gradients.
        decr_every_n_nan_or_inf(int): Decreases loss scaling every n 
                                      accumulated steps with nan or 
                                      inf gradients.
        incr_ratio(float): The multiplier to use when increasing the loss 
                           scaling.
        decr_ratio(float): The less-than-one-multiplier to use when decreasing 
                           the loss scaling.
        use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training 
        enabled.

    Examples:
	.. code-block:: python

	    loss = network()
            optimizer = fluid.optimizer.Adam(learning_rate=0.001)
	
            mp_optimizer = fluid.contrib.mixed_precision.decorate(
	              optimizer=optimizer, init_loss_scaling=8.0)
	
            ops, param_grads = mp_optimizer.minimize(loss)
            scaled_loss = mp_optimizer.get_loss_scaling()
    """
    if amp_lists is None:
        amp_lists = AutoMixedPrecisionLists()
    mp_optimizer = OptimizerWithMixedPrecison(
        optimizer, amp_lists, init_loss_scaling, use_dynamic_loss_scaling,
        incr_every_n_steps, decr_every_n_nan_or_inf, incr_ratio, decr_ratio)

    return mp_optimizer
