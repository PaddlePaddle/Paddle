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

from __future__ import print_function
from paddle.fluid import core, dygraph
from paddle.fluid.framework import _varbase_creator
from paddle.fluid.data_feeder import check_type
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import warnings
import numpy as np

__all__ = ['AmpScaler']


class AmpScaler(object):
    def __init__(
            self,
            enable=True,
            init_loss_scaling=2.**15,
            incr_ratio=2.0,
            decr_ratio=0.5,
            incr_every_n_steps=1000,
            decr_every_n_nan_or_inf=1, ):
        """
        :api_attr: imperative

        AmpScaler is used for Auto-Mixed-Precision training/inferring in imperative
        mode. It controls the scaling of loss, helps avoiding numerical overflow.
        The object of this class has two methods `scale()`, `minimize()`.

        `scale()` is used to multiply the loss by a scale ratio.
        `minimize()` is similar as `Optimizer.minimize()`, performs parameters updating.

        Commonly, it is used together with `amp_guard` to achieve Auto-Mixed-Precision in 
        imperative mode.

        Args:
            enable(bool, optional): Enable loss scaling or not. Default is True.
            init_loss_scaling (float, optional): The initial loss scaling factor. Default is 2**15.
            incr_ratio(float, optional): The multiplier to use when increasing the loss 
                            scaling. Default is 2.0.
            decr_ratio(float, optional): The less-than-one-multiplier to use when decreasing 
                            the loss scaling. Default is 0.5.
            incr_every_n_steps(int, optional): Increases loss scaling every n consecutive 
                                    steps with finite gradients. Default is 1000.
            decr_every_n_nan_or_inf(int, optional): Decreases loss scaling every n 
                                        accumulated steps with nan or inf gradients. Default is 2.
        """
        if enable and not core.is_compiled_with_cuda():
            warnings.warn(
                'Auto Mixed Precision can only be enabled with Paddle compiled with CUDA.'
            )
            self._enable = False
        else:
            self._enable = enable

        if self._enable:
            assert incr_ratio > 1.0, "The incr_ratio must be > 1.0."
            assert decr_ratio < 1.0, "The decr_ratio must be < 1.0."

            self._init_loss_scaling = init_loss_scaling
            self._incr_ratio = incr_ratio
            self._decr_ratio = decr_ratio
            self._incr_every_n_steps = incr_every_n_steps
            self._decr_every_n_nan_or_inf = decr_every_n_nan_or_inf
            self._incr_count = 0
            self._decr_count = 0

            self._found_inf = dygraph.to_variable(
                np.array([0]).astype(np.int32))
            self._scale = dygraph.to_variable(
                np.array([self._init_loss_scaling]).astype(np.float32))
            self._cache_founf_inf = None

    def scale(self, var):
        """
        Multiplies a variable(Tensor) by the scale factor and returns scaled outputs.  
        If this instance of :class:`AmpScaler` is not enabled, output are returned unmodified.

        Args:
            var (Variable):  The variable to scale.
        Returns:
            The scaled variable or original variable.
        """
        check_type(var, "var", core.VarBase, 'AmpScaler.scale()')

        if not self._enable:
            return var

        return var * self._scale

    def minimize(self, optimizer, *args, **kwargs):
        """
        Similar as `Optimizer.minimize()`, performs parameters updating.
        It first unscale the scaled gradients of paramers, then do one step parameter updating,
        and finally update loss scaling ratio. 

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
        """
        if not self._enable:
            return

        #  unscale the grad
        self._unscale(optimizer)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = False

        self._update()

    def _unscale(self, optimizer):
        if not self._enable:
            return
        inv_scale = 1.0 / self._scale
        param_grads = [
            param._grad_ivar() for param in optimizer._parameter_list
            if param._grad_ivar() is not None
        ]
        core.ops.amp_check_finite_and_scale(param_grads, inv_scale, param_grads,
                                            self._found_inf)

    def _update(self):
        """
        Updates the loss_scaling.
        """
        if not self._enable:
            return

        if self._cache_founf_inf:
            self._incr_count = 0
            self._decr_count = self._incr_count + 1
            if self._decr_count == self._decr_every_n_nan_or_inf:
                self._scale = self._scale * self._decr_ratio
                self._decr_count
            print('found infinite', float(self._scale))
        else:
            self._decr_count = 0
            self._incr_count = self._incr_count + 1
            if self._incr_count == self._incr_every_n_steps:
                self._scale = self._scale * self._incr_ratio
                self._incr_count = 0

        return
