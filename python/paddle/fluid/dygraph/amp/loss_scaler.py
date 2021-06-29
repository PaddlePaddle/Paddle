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
from paddle.fluid import core
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import _varbase_creator, _dygraph_tracer, dygraph_only
from paddle.fluid.data_feeder import check_type
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import warnings
import numpy as np

__all__ = ['AmpScaler']


class AmpScaler(object):
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
        use_dynamic_loss_scaling(bool, optional): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.
    Returns:
        An AmpScaler object.

    Examples:

     .. code-block:: python

        import numpy as np
        import paddle.fluid as fluid

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        with fluid.dygraph.guard():
            model = fluid.dygraph.Conv2D(3, 2, 3)
            optimizer = fluid.optimizer.SGDOptimizer(
                    learning_rate=0.01, parameter_list=model.parameters())
            scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
            data = fluid.dygraph.to_variable(data)
            with fluid.dygraph.amp_guard():
                conv = model(data)
                loss = fluid.layers.reduce_mean(conv)
                scaled = scaler.scale(loss)
                scaled.backward()
                scaler.minimize(optimizer, scaled)         
    """

    @dygraph_only
    def __init__(self,
                 enable=True,
                 init_loss_scaling=2.**15,
                 incr_ratio=2.0,
                 decr_ratio=0.5,
                 incr_every_n_steps=1000,
                 decr_every_n_nan_or_inf=1,
                 use_dynamic_loss_scaling=True):

        tracer = _dygraph_tracer()
        if not tracer:
            raise ValueError(
                "current_tracer is None, maybe it is not in imperative mode.")

        if enable and not (tracer._expected_place.is_gpu_place() or
                           tracer._expected_place.is_xpu_place()):
            warnings.warn(
                'AmpScaler can only be enabled on CUDAPlace and XPUPlace, current place is %s, so it makes no effect.'
                % tracer._expected_place)
            enable = False

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
            self._use_dynamic_loss_scaling = use_dynamic_loss_scaling

            self._found_inf = to_variable(np.array([0]).astype(np.bool))
            self._scale = to_variable(
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
        
        Examples:
            .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            with fluid.dygraph.guard():
                model = fluid.dygraph.Conv2D(3, 2, 3)
                optimizer = fluid.optimizer.SGDOptimizer(
                        learning_rate=0.01, parameter_list=model.parameters())
                scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
                data = fluid.dygraph.to_variable(data)
                with fluid.dygraph.amp_guard():
                    conv = model(data)
                    loss = fluid.layers.reduce_mean(conv)
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.minimize(optimizer, scaled) 
        """
        check_type(var, "var", core.VarBase, 'AmpScaler.scale()')

        if not self._enable:
            return var

        return var * self._scale

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `Optimizer.minimize()`, which performs parameters updating.
        
        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `Optimizer.minimize()`.

        Examples:
            .. code-block:: python

            import numpy as np
            import paddle.fluid as fluid

            data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
            with fluid.dygraph.guard():
                model = fluid.dygraph.Conv2D(3, 2, 3)
                optimizer = fluid.optimizer.SGDOptimizer(
                        learning_rate=0.01, parameter_list=model.parameters())
                scaler = fluid.dygraph.AmpScaler(init_loss_scaling=1024)
                data = fluid.dygraph.to_variable(data)
                with fluid.dygraph.amp_guard():
                    conv = model(data)
                    loss = fluid.layers.reduce_mean(conv)
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.minimize(optimizer, scaled) 
        """
        if not self._enable:
            return optimizer.minimize(*args, **kwargs)

        #  unscale the grad
        self._unscale(optimizer)

        optimize_ops, params_grads = (None, None)

        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            # uopdate the scale
            self._update()

        return optimize_ops, params_grads

    def _unscale(self, optimizer):
        if not self._enable:
            return
        param_grads = [
            param._grad_ivar() for param in optimizer._parameter_list
            if param._grad_ivar() is not None
        ]
        core.ops.check_finite_and_unscale(param_grads, self._scale, param_grads,
                                          self._found_inf)

    def _update(self):
        """
        Updates the loss_scaling.
        """
        if not self._enable:
            return

        if self._cache_founf_inf:
            self._incr_count = 0
            self._decr_count = self._decr_count + 1
            if self._decr_count == self._decr_every_n_nan_or_inf:
                print(
                    'Found inf or nan, current scale is: {}, decrease to: {}*{}'.
                    format(
                        float(self._scale),
                        float(self._scale), float(self._decr_ratio)))
                self._scale = self._scale * self._decr_ratio
                self._decr_count = 0
        else:
            self._decr_count = 0
            self._incr_count = self._incr_count + 1
            if self._incr_count == self._incr_every_n_steps:
                self._scale = self._scale * self._incr_ratio
                self._incr_count = 0

        return
