#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import contextlib
from . import layers
from . import framework
from . import core

__all__ = ['WeightDecay', ]


@contextlib.contextmanager
def append_weight_decay(param_and_grads, weight_decay=None):
    """Add decay for the weight.

    Appends weight decay operators in the BlockDesc.
    This will update the optimized parameters by using the
    parameters before optimization.

    Args:
        param_and_grads: A list of (parameters, gradients) pairs
            that need to be updated.
        weight_decay(WeightDecayBase): A WeightDecay Objection, such as
            fluid.weight_decay.WeightDecay.

    Raises:
        Exception: Unknown weight decay type.
    """

    if weight_decay is not None:
        weight_decay(param_and_grads)
    yield
    if weight_decay is not None:
        weight_decay.apply()


class WeightDecayBase(object):
    def __str__(self):
        raise NotImplementedError()

    def __call__(self, params_and_grads):
        raise NotImplementedError()

    def apply(self):
        raise NotImplementedError()


class WeightDecay(WeightDecayBase):
    """
    WeightDecay is used to update the optimized parameters by using the
    parameters before optimization.

    Args:
        coeff (float|Variable): The maximum value to clip by.
        attempt_decay_param_fun (function|None): If it is not None,
            only variables that makes attempt_decay_param_fun(variable)==True
            will be updated. It only works when we want to specify variables.
            Default: None.

    Examples:
        .. code-block:: python
            learning_rate = 0.1
            optimizer = fluid.optimizer.Adagrad(
                learning_rate=learning_rate,
                weight_decay=fluid.weight_decay.WeightDecay(
                    coeff=learning_rate))
    """

    def __init__(self, coeff=0.0, attempt_decay_param_fun=None):

        if not isinstance(coeff, float) and \
                not isinstance(coeff, framework.Variable):
            raise TypeError("coeff should be float or Variable.")

        self.scaled_params_ = dict()
        self.params_name_ = []
        self.attempt_decay_param_fun_ = attempt_decay_param_fun
        self.coeff_ = coeff

    def __call__(self, params_and_grads):
        if self.coeff_ == 0.0:
            return
        for param, grad in params_and_grads:
            # If no gradient then we don't need to do anything
            if grad is None:
                continue
            if self.attempt_decay_param_fun_ is not None \
                    and not self.attempt_decay_param_fun_(param.name):
                continue

            coeff = self.coeff_
            if isinstance(coeff, float) and \
                            param.dtype is not core.VarDesc.VarType.FP32:
                dtype = framework.convert_dtype_np_dtype_to_(param.dtype)
                coeff = dtype(self.coeff_)

            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                assert param.name not in self.params_name_
                self.scaled_params_[param.name] = (param, grad, param * coeff)
                self.params_name_.append(param.name)

    def apply(self):
        """
        Update the optimized parameters.
        """
        if self.coeff_ == 0.0:
            return
        for p_name in self.scaled_params_.keys():
            param, grad, scaled_param = self.scaled_params_[p_name]
            with param.block.program._optimized_guard(
                [param, grad]), framework.name_scope('weight decay'):
                layers.elementwise_sub(x=param, y=scaled_param, out=param)

    def __str__(self):
        info = "Weight Decay, params: "
        for p in self.params_name_:
            info += p
            info += ", "

        return info


WeightDecay = WeightDecay
