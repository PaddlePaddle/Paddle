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

import warnings
from collections import defaultdict
from enum import Enum

import numpy as np

from paddle import _C_ops, _legacy_C_ops
from paddle.fluid import core
from paddle.fluid.data_feeder import check_type
from paddle.fluid.dygraph import to_variable
from paddle.fluid.framework import _dygraph_tracer, dygraph_only
from paddle.framework import in_dynamic_mode

from .auto_cast import amp_global_state


class OptimizerState(Enum):
    INIT = 0
    UNSCALED = 1
    STEPPED = 2


def _refresh_optimizer_state():
    return {"state": OptimizerState.INIT}


class AmpScaler:
    """
    AmpScaler is used for Auto-Mixed-Precision training/inferring in imperative
    mode. It controls the scaling of loss, helps avoiding numerical overflow.
    The object of this class has seventeen methods `scale()`, `unscale_()`, `minimize()` and `get`/`set` api of parameters.

    `scale()` is used to multiply the loss by a scale ratio.
    `unscale_()` is used to unscale the gradients of parameters, multiplies the gradients of parameters by 1/(scale ratio)
    `minimize()` is similar as `optimizer.minimize()`, performs parameters updating, and it will update the loss_scaling.

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
        import paddle

        data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
        model = paddle.nn.Conv2D(3, 2, 3)
        optimizer = paddle.optimizer.SGDOptimizer(
                learning_rate=0.01, parameter_list=model.parameters())
        scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
        data = paddle.to_tensor(data)
        with paddle.amp.amp_guard():
            conv = model(data)
            loss = paddle.mean(conv)
            scaled = scaler.scale(loss)
            scaled.backward()
            scaler.minimize(optimizer, scaled)
    """

    @dygraph_only
    def __init__(
        self,
        enable=True,
        init_loss_scaling=2.0**15,
        incr_ratio=2.0,
        decr_ratio=0.5,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=1,
        use_dynamic_loss_scaling=True,
    ):

        tracer = _dygraph_tracer()
        if not tracer:
            raise ValueError(
                "current_tracer is None, maybe it is not in imperative mode."
            )

        if enable and not (
            tracer._expected_place.is_gpu_place()
            or tracer._expected_place.is_xpu_place()
            or tracer._expected_place.is_custom_place()
        ):
            warnings.warn(
                'AmpScaler can only be enabled on CUDAPlace, XPUPlace and CustomPlace, current place is %s, so it makes no effect.'
                % tracer._expected_place
            )
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

            self._found_inf = to_variable(np.array([0]).astype(np.bool_))
            self._temp_found_inf_value_false = to_variable(
                np.array([0]).astype(np.bool_)
            )
            self._temp_found_inf_fp16 = to_variable(
                np.array([0]).astype(np.bool_)
            )
            self._temp_found_inf_bf16 = to_variable(
                np.array([0]).astype(np.bool_)
            )
            self._temp_found_inf_fp32 = to_variable(
                np.array([0]).astype(np.bool_)
            )
            self._scale = to_variable(
                np.array([self._init_loss_scaling]).astype(np.float32)
            )
            self._cache_founf_inf = None
            self._optimizer_states = defaultdict(_refresh_optimizer_state)

    def scale(self, var):
        """
        Multiplies a Tensor by the scale factor and returns scaled outputs.
        If this instance of :class:`AmpScaler` is not enabled, output are returned unmodified.

        Args:
            var (Tensor):  The Tensor to scale.
        Returns:
            The scaled Tensor or original Tensor.

        Examples:

            .. code-block:: python

                import numpy as np
                import paddle

                data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
                model = paddle.nn.Conv2D(3, 2, 3)
                optimizer = paddle.optimizer.SGDOptimizer(
                        learning_rate=0.01, parameter_list=model.parameters())
                scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
                data = paddle.to_tensor(data)
                with paddle.amp.amp_guard():
                    conv = model(data)
                    loss = paddle.mean(conv)
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.minimize(optimizer, scaled)
        """
        check_type(var, "var", core.eager.Tensor, 'AmpScaler.scale()')

        if (
            self._enable
            and amp_global_state().amp_dtype != 'float16'
            and self._use_dynamic_loss_scaling
        ):
            self._enable = False
            self._use_dynamic_loss_scaling = False
            warnings.warn(
                'It is not recommended to use dynamic loss scaling for %s, so GradScaler is disable by default.'
                % (amp_global_state().amp_dtype)
            )

        if not self._enable:
            return var

        return var * self._scale

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `Optimizer.minimize()`, which performs parameters updating.

        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `Optimizer.minimize()`.

        Examples:

            .. code-block:: python

                import numpy as np
                import paddle

                data = np.random.uniform(-1, 1, [10, 3, 32, 32]).astype('float32')
                model = paddle.nn.Conv2D(3, 2, 3)
                optimizer = paddle.optimizer.SGDOptimizer(
                        learning_rate=0.01, parameter_list=model.parameters())
                scaler = paddle.amp.AmpScaler(init_loss_scaling=1024)
                data = paddle.to_tensor(data)
                with paddle.amp.amp_guard():
                    conv = model(data)
                    loss = paddle.mean(conv)
                    scaled = scaler.scale(loss)
                    scaled.backward()
                    scaler.minimize(optimizer, scaled)
        """
        if not self._enable:
            return optimizer.minimize(*args, **kwargs)

        optimizer_state = self._optimizer_states[id(optimizer)]

        #  unscale the grad
        if optimizer_state["state"] is OptimizerState.INIT:
            self._unscale(optimizer)

        optimize_ops, params_grads = (None, None)

        if hasattr(optimizer, "_set_auxiliary_var"):
            optimizer._set_auxiliary_var('found_inf', self._found_inf)
            optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = optimizer._get_auxiliary_var('found_inf')
        else:
            if self._found_inf:
                self._cache_founf_inf = True
            else:
                optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
                self._cache_founf_inf = False

        if self._use_dynamic_loss_scaling:
            # uopdate the scale
            self._update()

        self._optimizer_states = defaultdict(_refresh_optimizer_state)

        return optimize_ops, params_grads

    def _unscale(self, optimizer):
        """
        Unscale the gradients of parameters, multiplies the gradients of parameters by 1/(loss scaling ratio).
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.
        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
        Returns:
            The unscaled parameters or original parameters.
        """
        if not self._enable:
            return

        optimizer_state = self._optimizer_states[id(optimizer)]

        if optimizer_state["state"] is OptimizerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        elif optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        if getattr(optimizer, '_param_groups', None) and isinstance(
            optimizer._param_groups[0], dict
        ):
            param_grads = []
            param_grads_fp16 = []
            param_grads_bf16 = []
            param_grads_fp32 = []
            for group in optimizer._param_groups:
                for param in group['params']:
                    if param._grad_ivar() is not None:
                        param_grads.append(param._grad_ivar())
                        if (
                            param._grad_ivar().dtype
                            == core.VarDesc.VarType.FP16
                        ):
                            param_grads_fp16.append(param._grad_ivar())
                        elif (
                            param._grad_ivar().dtype
                            == core.VarDesc.VarType.BF16
                        ):
                            param_grads_bf16.append(param._grad_ivar())
                        else:
                            param_grads_fp32.append(param._grad_ivar())
        else:
            if in_dynamic_mode():
                # It is very time-consuming to call c++ functions in a loop on the python side.
                # We put this part of the code on the c++ side to improve the speed in eager mode.
                (
                    param_grads_fp16,
                    param_grads_bf16,
                    param_grads_fp32,
                ) = core.eager.get_grads_lists(optimizer._parameter_list)
            else:
                # Keep the original code to support legacy mode.
                # Delete the else branch when the legacy mode exits.
                param_grads = [
                    param._grad_ivar()
                    for param in optimizer._parameter_list
                    if param._grad_ivar() is not None
                ]
                param_grads_fp16 = [
                    param
                    for param in param_grads
                    if param.dtype == core.VarDesc.VarType.FP16
                ]
                param_grads_bf16 = [
                    param
                    for param in param_grads
                    if param.dtype == core.VarDesc.VarType.BF16
                ]
                param_grads_fp32 = [
                    param
                    for param in param_grads
                    if param.dtype == core.VarDesc.VarType.FP32
                ]
        self._found_inf = self._temp_found_inf_value_false
        if len(param_grads_fp16):
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_fp16,
                self._scale,
                param_grads_fp16,
                self._temp_found_inf_fp16,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, self._temp_found_inf_fp16
            )
        if len(param_grads_bf16):
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_bf16,
                self._scale,
                param_grads_bf16,
                self._temp_found_inf_bf16,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, self._temp_found_inf_bf16
            )
        if len(param_grads_fp32):
            _legacy_C_ops.check_finite_and_unscale(
                param_grads_fp32,
                self._scale,
                param_grads_fp32,
                self._temp_found_inf_fp32,
            )
            self._found_inf = _C_ops.bitwise_or(
                self._found_inf, self._temp_found_inf_fp32
            )

        optimizer_state["state"] = OptimizerState.UNSCALED

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
                    'Found inf or nan, current scale is: {}, decrease to: {}*{}'.format(
                        float(self._scale),
                        float(self._scale),
                        float(self._decr_ratio),
                    )
                )
                self._scale = self._scale * self._decr_ratio
                self._decr_count = 0
        else:
            self._decr_count = 0
            self._incr_count = self._incr_count + 1
            if self._incr_count == self._incr_every_n_steps:
                self._scale = self._scale * self._incr_ratio
                self._incr_count = 0

        return

    def is_enable(self):
        """
        Enable loss scaling or not.

        Returns:
            bool: enable loss scaling return True else return False.
        """
        return self._enable

    def is_use_dynamic_loss_scaling(self):
        """
        Whether to use dynamic loss scaling.

        Returns:
            bool: if fixed loss_scaling is used return False, if the loss scaling is updated dynamicly return true.
        """
        return self._use_dynamic_loss_scaling

    def get_init_loss_scaling(self):
        """
        Return the initial loss scaling factor.

        Reurns:
            float:  the initial loss scaling factor.
        """
        return self._init_loss_scaling

    def set_init_loss_scaling(self, new_init_loss_scaling):
        """
        Set the initial loss scaling factor by `new_init_loss_scaling`.

        Args:
            new_init_loss_scaling(int):  The new_init_loss_scaling used to update initial loss scaling factor.s
        """
        self._init_loss_scaling = new_init_loss_scaling
        self._scale = to_variable(
            np.array([self._init_loss_scaling]).astype(np.float32)
        )

    def get_incr_ratio(self):
        """
        Return the multiplier to use when increasing the loss scaling.

        Reurns:
            float:  the multiplier to use when increasing the loss scaling.
        """
        return self._incr_ratio

    def set_incr_ratio(self, new_incr_ratio):
        """
        Set the multiplier to use when increasing the loss scaling by `new_incr_ratio`, `new_incr_ratio` should > 1.0.

        Args:
            new_incr_ratio(float):  The new_incr_ratio used to update the multiplier to use when increasing the loss scaling.
        """
        assert new_incr_ratio > 1.0, "The new_incr_ratio must be > 1.0."
        self._incr_ratio = new_incr_ratio

    def get_decr_ratio(self):
        """
        Get the less-than-one-multiplier to use when decreasing the loss scaling.

        Reurns:
            float:  the less-than-one-multiplier to use when decreasing the loss scaling.
        """
        return self._decr_ratio

    def set_decr_ratio(self, new_decr_ratio):
        """
        Set the less-than-one-multiplier to use when decreasing the loss scaling by `new_incr_ratio`, `new_decr_ratio` should < 1.0.

        Args:
            new_decr_ratio(float):  The new_decr_ratio used to update the less-than-one-multiplier to use when decreasing the loss scaling.
        """
        assert new_decr_ratio < 1.0, "The new_decr_ratio must be < 1.0."
        self._decr_ratio = new_decr_ratio

    def get_incr_every_n_steps(self):
        """
        Return the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Reurns:
            int:  the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        """
        return self._incr_every_n_steps

    def set_incr_every_n_steps(self, new_incr_every_n_steps):
        """
        Set the num `n` by `new_incr_every_n_steps`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Args:
            new_incr_every_n_steps(int):  The new_incr_every_n_steps used to update the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.
        """
        self._incr_every_n_steps = new_incr_every_n_steps

    def get_decr_every_n_nan_or_inf(self):
        """
        Return the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Reurns:
            int:  the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        """
        return self._decr_every_n_nan_or_inf

    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf):
        """
        Set the num `n` by `new_decr_every_n_nan_or_inf`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Args:
            new_decr_every_n_nan_or_inf(int):  The new_decr_every_n_nan_or_inf used to update the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.
        """
        self._decr_every_n_nan_or_inf = new_decr_every_n_nan_or_inf

    def state_dict(self):
        """
        Returns the state of the scaler as a `dict`, If this instance is not enabled, returns an empty dict.

        Reurns:
            A dict of scaler includes:
            scale (tensor): The loss scaling factor.
            incr_ratio(float): The multiplier to use when increasing the loss scaling.
            decr_ratio(float): The less-than-one-multiplier to use when decreasing the loss scaling.
            incr_every_n_steps(int): Increases loss scaling every n consecutive steps with finite gradients.
            decr_every_n_nan_or_inf(int): Decreases loss scaling every n accumulated steps with nan or inf gradients.
            incr_count(int): The number of recent consecutive unskipped steps.
            decr_count(int): The number of recent consecutive skipped steps.
            use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.
        """
        return (
            {
                "scale": self._scale.numpy(),
                "incr_ratio": self._incr_ratio,
                "decr_ratio": self._decr_ratio,
                "incr_every_n_steps": self._incr_every_n_steps,
                "decr_every_n_nan_or_inf": self._decr_every_n_nan_or_inf,
                "incr_count": self._incr_count,
                "decr_count": self._decr_count,
                "use_dynamic_loss_scaling": self._use_dynamic_loss_scaling,
            }
            if self._enable
            else {}
        )

    def load_state_dict(self, state_dict):
        """
        Loads the scaler state.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to `AmpScaler.state_dict()`.
        """
        if not self._enable:
            return

        if len(state_dict) == 0:
            raise RuntimeError(
                "The input state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        self._init_loss_scaling = state_dict["scale"][0]
        self._scale = to_variable(
            np.array([self._init_loss_scaling]).astype(np.float32)
        )
        self._incr_ratio = state_dict["incr_ratio"]
        self._decr_ratio = state_dict["decr_ratio"]
        self._incr_every_n_steps = state_dict["incr_every_n_steps"]
        self._decr_every_n_nan_or_inf = state_dict["decr_every_n_nan_or_inf"]
        self._incr_count = state_dict["incr_count"]
        self._decr_count = state_dict["decr_count"]
        self._use_dynamic_loss_scaling = state_dict["use_dynamic_loss_scaling"]


class GradScaler(AmpScaler):
    """
    GradScaler is used for Auto-Mixed-Precision training in dynamic graph mode.
    It controls the scaling of loss, helps avoiding numerical overflow.
    The object of this class has nineteen methods `scale()`, `unscale_()`, `minimize()`, `step()`, `update()` and `get`/`set` api of parameters.

    `scale()` is used to multiply the loss by a scale ratio.
    `unscale_()` is used to unscale the gradients of parameters, multiplies the gradients of parameters by 1/(scale ratio)
    `minimize()` is similar as `optimizer.minimize()`, performs parameters updating, and it will update the loss_scaling, it equal to `step()` + `update()`.
    `step()` is similar as `optimizer.step()`, which performs parameters updating.
    `update` is used to update the loss_scaling.


    Commonly, it is used together with `paddle.amp.auto_cast` to achieve Auto-Mixed-Precision in
    dynamic graph mode.

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
        An GradScaler object.

    Examples:

        .. code-block:: python

            import paddle

            model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
            optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
            scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
            data = paddle.rand([10, 3, 32, 32])

            with paddle.amp.auto_cast():
                conv = model(data)
                loss = paddle.mean(conv)

            scaled = scaler.scale(loss)  # scale the loss
            scaled.backward()            # do backward
            scaler.minimize(optimizer, scaled)  # update parameters
            optimizer.clear_grad()
    """

    def __init__(
        self,
        enable=True,
        init_loss_scaling=2.0**15,
        incr_ratio=2.0,
        decr_ratio=0.5,
        incr_every_n_steps=1000,
        decr_every_n_nan_or_inf=2,
        use_dynamic_loss_scaling=True,
    ):
        super().__init__(
            enable,
            init_loss_scaling,
            incr_ratio,
            decr_ratio,
            incr_every_n_steps,
            decr_every_n_nan_or_inf,
            use_dynamic_loss_scaling,
        )

    def scale(self, var):
        """
        Multiplies a Tensor by the scale factor and returns scaled outputs.
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            var (Tensor):  The tensor to scale.
        Returns:
            The scaled tensor or original tensor.

        Examples:

            .. code-block:: python

                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters
                optimizer.clear_grad()
        """
        return super().scale(var)

    def minimize(self, optimizer, *args, **kwargs):
        """
        This function is similar as `optimizer.minimize()`, which performs parameters updating.

        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Finally, the loss scaling ratio is updated.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.
            args:  Arguments, which will be forward to `optimizer.minimize()`.
            kwargs: Keyword arguments, which will be forward to `optimizer.minimize()`.

        Examples:

            .. code-block:: python

                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])

                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)

                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()            # do backward
                scaler.minimize(optimizer, scaled)  # update parameters
                optimizer.clear_grad()
        """
        return super().minimize(optimizer, *args, **kwargs)

    def step(self, optimizer):
        """
        This function is similar as `optimizer.step()`, which performs parameters updating.

        If the scaled gradients of parameters contains NAN or INF, the parameters updating is skipped.
        Otherwise, if `unscale_()` has not been called, it first unscales the scaled gradients of parameters, then updates the parameters.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.

        Examples:

            .. code-block:: python

                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()            # do backward
                scaler.step(optimizer)       # update parameters
                scaler.update()              # update the loss scaling ratio
                optimizer.clear_grad()
        """
        if not self._enable:
            return optimizer.step()

        optimizer_state = self._optimizer_states[id(optimizer)]
        if optimizer_state["state"] is OptimizerState.STEPPED:
            raise RuntimeError(
                "step() has already been called since the last update()."
            )

        #  unscale the grad
        if optimizer_state["state"] is OptimizerState.INIT:
            self._unscale(optimizer)

        if hasattr(optimizer, "_set_auxiliary_var"):
            optimizer._set_auxiliary_var('found_inf', self._found_inf)
            optimizer.step()
            self._cache_founf_inf = optimizer._get_auxiliary_var('found_inf')
        else:
            if self._found_inf:
                self._cache_founf_inf = True
            else:
                optimizer.step()
                self._cache_founf_inf = False

        optimizer_state["state"] = OptimizerState.STEPPED

        if not self._use_dynamic_loss_scaling:
            self._optimizer_states = defaultdict(_refresh_optimizer_state)

    def update(self):
        """
        Updates the loss_scaling.

        Examples:

            .. code-block:: python

                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)     # scale the loss
                scaled.backward()               # do backward
                scaler.step(optimizer)          # update parameters
                scaler.update()                 # update the loss scaling ratio
                optimizer.clear_grad()
        """
        if not self._enable:
            return
        if self._use_dynamic_loss_scaling:
            self._update()
            self._optimizer_states = defaultdict(_refresh_optimizer_state)
        return

    def unscale_(self, optimizer):
        """
        Unscale the gradients of parameters, multiplies the gradients of parameters by 1/(loss scaling ratio).
        If this instance of :class:`GradScaler` is not enabled, output are returned unmodified.

        Args:
            optimizer(Optimizer):  The optimizer used to update parameters.

        Returns:
            The unscaled parameters or original parameters.

        Examples:

            .. code-block:: python

                # required: gpu
                import paddle

                model = paddle.nn.Conv2D(3, 2, 3, bias_attr=True)
                optimizer = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())
                scaler = paddle.amp.GradScaler(init_loss_scaling=1024)
                data = paddle.rand([10, 3, 32, 32])
                with paddle.amp.auto_cast():
                    conv = model(data)
                    loss = paddle.mean(conv)
                scaled = scaler.scale(loss)  # scale the loss
                scaled.backward()            # do backward
                scaler.unscale_(optimizer)    # unscale the parameter
                scaler.step(optimizer)
                scaler.update()
                optimizer.clear_grad()
        """
        return super()._unscale(optimizer)

    def is_enable(self):
        """
        Enable loss scaling or not.

        Returns:
            bool: enable loss scaling return True else return False.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                enable = scaler.is_enable()
                print(enable) # True
        """
        return super().is_enable()

    def is_use_dynamic_loss_scaling(self):
        """
        Whether to use dynamic loss scaling.

        Returns:
            bool: if fixed loss_scaling is used return False, if the loss scaling is updated dynamicly return true.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                use_dynamic_loss_scaling = scaler.is_use_dynamic_loss_scaling()
                print(use_dynamic_loss_scaling) # True
        """
        return super().is_use_dynamic_loss_scaling()

    def get_init_loss_scaling(self):
        """
        Return the initial loss scaling factor.

        Reurns:
            float:  the initial loss scaling factor.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                init_loss_scaling = scaler.get_init_loss_scaling()
                print(init_loss_scaling) # 1024
        """
        return super().get_init_loss_scaling()

    def set_init_loss_scaling(self, new_init_loss_scaling):
        """
        Set the initial loss scaling factor by `new_init_loss_scaling`.

        Args:
            new_init_loss_scaling(float):  The new_init_loss_scaling used to update initial loss scaling factor.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_init_loss_scaling()) # 1024
                new_init_loss_scaling = 1000
                scaler.set_init_loss_scaling(new_init_loss_scaling)
                print(scaler.get_init_loss_scaling()) # 1000
        """
        super().set_init_loss_scaling(new_init_loss_scaling)

    def get_incr_ratio(self):
        """
        Return the multiplier to use when increasing the loss scaling.

        Reurns:
            float:  the multiplier to use when increasing the loss scaling.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_ratio = scaler.get_incr_ratio()
                print(incr_ratio) # 2.0
        """
        return super().get_incr_ratio()

    def set_incr_ratio(self, new_incr_ratio):
        """
        Set the multiplier to use when increasing the loss scaling by `new_incr_ratio`, `new_incr_ratio` should > 1.0.

        Args:
            new_incr_ratio(float):  The new_incr_ratio used to update the multiplier to use when increasing the loss scaling.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_incr_ratio()) # 2.0
                new_incr_ratio = 3.0
                scaler.set_incr_ratio(new_incr_ratio)
                print(scaler.get_incr_ratio()) # 3.0
        """
        super().set_incr_ratio(new_incr_ratio)

    def get_decr_ratio(self):
        """
        Get the less-than-one-multiplier to use when decreasing the loss scaling.

        Reurns:
            float:  the less-than-one-multiplier to use when decreasing the loss scaling.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_ratio = scaler.get_decr_ratio()
                print(decr_ratio) # 0.5
        """
        return super().get_decr_ratio()

    def set_decr_ratio(self, new_decr_ratio):
        """
        Set the less-than-one-multiplier to use when decreasing the loss scaling by `new_incr_ratio`, `new_decr_ratio` should < 1.0.

        Args:
            new_decr_ratio(float):  The new_decr_ratio used to update the less-than-one-multiplier to use when decreasing the loss scaling.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_decr_ratio()) # 0.5
                new_decr_ratio = 0.1
                scaler.set_decr_ratio(new_decr_ratio)
                print(scaler.get_decr_ratio()) # 0.1
        """
        super().set_decr_ratio(new_decr_ratio)

    def get_incr_every_n_steps(self):
        """
        Return the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Reurns:
            int:  the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                incr_every_n_steps = scaler.get_incr_every_n_steps()
                print(incr_every_n_steps) # 1000
        """
        return super().get_incr_every_n_steps()

    def set_incr_every_n_steps(self, new_incr_every_n_steps):
        """
        Set the num `n` by `new_incr_every_n_steps`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Args:
            new_incr_every_n_steps(int):  The new_incr_every_n_steps used to update the num `n`, `n` represent increases loss scaling every `n` consecutive steps with finite gradients.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_incr_every_n_steps()) # 1000
                new_incr_every_n_steps = 2000
                scaler.set_incr_every_n_steps(new_incr_every_n_steps)
                print(scaler.get_incr_every_n_steps()) # 2000
        """
        super().set_incr_every_n_steps(new_incr_every_n_steps)

    def get_decr_every_n_nan_or_inf(self):
        """
        Return the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Reurns:
            int:  the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                decr_every_n_nan_or_inf = scaler.get_decr_every_n_nan_or_inf()
                print(decr_every_n_nan_or_inf) # 2
        """
        return super().get_decr_every_n_nan_or_inf()

    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf):
        """
        Set the num `n` by `new_decr_every_n_nan_or_inf`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Args:
            new_decr_every_n_nan_or_inf(int):  The new_decr_every_n_nan_or_inf used to update the num `n`, `n` represent decreases loss scaling every `n` accumulated steps with nan or inf gradients.

        Examples:
            .. code-block:: python

                # required: gpu,xpu
                import paddle
                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                print(scaler.get_decr_every_n_nan_or_inf()) # 2
                new_decr_every_n_nan_or_inf = 3
                scaler.set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)
                print(scaler.get_decr_every_n_nan_or_inf()) # 3
        """
        super().set_decr_every_n_nan_or_inf(new_decr_every_n_nan_or_inf)

    def state_dict(self):
        """
        Returns the state of the scaler as a `dict`, If this instance is not enabled, returns an empty dict.

        Reurns:
            A dict of scaler includes:
            scale (tensor): The loss scaling factor.
            incr_ratio(float): The multiplier to use when increasing the loss scaling.
            decr_ratio(float): The less-than-one-multiplier to use when decreasing the loss scaling.
            incr_every_n_steps(int): Increases loss scaling every n consecutive steps with finite gradients.
            decr_every_n_nan_or_inf(int): Decreases loss scaling every n accumulated steps with nan or inf gradients.
            incr_count(int): The number of recent consecutive unskipped steps.
            decr_count(int): The number of recent consecutive skipped steps.
            use_dynamic_loss_scaling(bool): Whether to use dynamic loss scaling. If False, fixed loss_scaling is used. If True, the loss scaling is updated dynamicly. Default is True.


        Examples:

            .. code-block:: python

                # required: gpu,xpu
                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                scaler_state = scaler.state_dict()
        """
        return super().state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the scaler state.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to `GradScaler.state_dict()`.

        Examples:

            .. code-block:: python

                # required: gpu,xpu
                import paddle

                scaler = paddle.amp.GradScaler(enable=True,
                                               init_loss_scaling=1024,
                                               incr_ratio=2.0,
                                               decr_ratio=0.5,
                                               incr_every_n_steps=1000,
                                               decr_every_n_nan_or_inf=2,
                                               use_dynamic_loss_scaling=True)
                scaler_state = scaler.state_dict()
                scaler.load_state_dict(scaler_state)
        """
        super().load_state_dict(state_dict)
