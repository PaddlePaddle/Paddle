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
from .fp16_utils import create_master_params_grads, master_param_to_train_param

__all__ = ["decorate"]


class OptimizerWithMixedPrecison(object):
    """
    Optimizer class with mixed-precision training.

    Args:
        optimizer (Optimizer): A common Optimizer object.
        init_loss_scaling (float): The initial loss scaling factor.
        use_dynamic_loss_scaling (bool): Whether to use dynamic loss scaling.
    """

    def __init__(self, optimizer, init_loss_scaling, use_dynamic_loss_scaling):
        self._optimizer = optimizer
        self._param_grads = None
        self._train_program = default_main_program()
        self._startup_prog = default_startup_program()
        self._loss_scaling = init_loss_scaling
        self._use_dynamic_loss_scaling = use_dynamic_loss_scaling

        # Ensure the data type of learning rate vars is float32
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
            A list of tuple (param, grad), which are a parameter and its gradient
            respectively, and the scaled loss.
        """
        scaled_loss = loss * self._loss_scaling
        self._param_grads = self._optimizer.backward(
            scaled_loss, startup_program, parameter_list, no_grad_set,
            callbacks)
        master_params_grads = create_master_params_grads(
            self._param_grads, self._train_program, self._startup_prog,
            self._loss_scaling)

        return master_params_grads, scaled_loss

    def apply_gradients(self, master_params_grads):
        """
        Update master parameters by their gradients, and cast to parameters
        in float16.
  
        Args:
            master_params_grads (list): A list of master params and grads.
    
        Returns:
            A list of optimize operators.
        """
        optimize_ops = self._optimizer.apply_gradients(master_params_grads)
        master_param_to_train_param(master_params_grads, self._param_grads,
                                    self._train_program)
        return optimize_ops

    def minimize(self, loss):
        """
        Perform optimization by minimizing the given loss.

        Args:
            loss (Variable): The loss Variable.

        Returns:
            The scaled loss by scaling factor, the list of optimize ops, and a
            list of master parameters and gradients.
        """
        master_params_grads, scaled_loss = self.backward(loss)
        optimize_ops = self.apply_gradients(master_params_grads)

        return scaled_loss, optimize_ops, master_params_grads


def decorate(optimizer, init_loss_scaling=1.0, use_dynamic_loss_scaling=False):
    """ 
    Decorate the given optimizer to adapt to the mixed-precision training.

    Args:
        optimizer(Optimizer): A common Optimizer.
        init_loss_scaling(float): The initial loss scaling factor.
        use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling.

    Returns:
        An optimizer acting like a normal one but with mixed-precision training 
        enabled.
    """

    mp_optimizer = OptimizerWithMixedPrecison(optimizer, init_loss_scaling,
                                              use_dynamic_loss_scaling)

    return mp_optimizer
