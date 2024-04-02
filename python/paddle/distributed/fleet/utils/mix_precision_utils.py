# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


from collections import defaultdict
from types import MethodType

import numpy as np

import paddle
from paddle import _legacy_C_ops, nn
from paddle.base import framework
from paddle.base.dygraph import (
    base as imperative_base,
)
from paddle.distributed import fleet
from paddle.distributed.fleet.utils.hybrid_parallel_util import (
    obtain_optimizer_parameters_list,
)
from paddle.framework import core
from paddle.utils import deprecated


class MixPrecisionLayer(nn.Layer):
    def __init__(self, layers, dtype="float16"):
        super().__init__(layers.full_name() + "_mix_precision")

        self._layers = layers
        self._dtype = dtype

        assert self._dtype in ["float16", "bfloat16"]

        for param in self._layers.parameters():
            if not hasattr(param, "main_grad"):
                param.main_grad = None
                param._register_grad_hook(self._update_main_grad_hook(param))

    def _update_main_grad_hook(self, param):
        """Create the update_main_grad hook for back-prop."""

        # Hook used for back-prop and grad-merge.
        @paddle.autograd.no_grad()
        def param_hook(tmp_grad):
            assert (
                param.grad is None
            ), f"In main_grad node, param.grad should be None, but find param[{param.name}] has grad."
            if tmp_grad is not None and tmp_grad._is_initialized():
                # Some previous pylayer may return None, should check grad validation.
                if param.main_grad is None:
                    param.main_grad = core.eager.Tensor(
                        value=tmp_grad.cast(paddle.float32).value(),
                        place=tmp_grad.place,
                        name="main_grad@" + param.name,
                    )
                else:
                    param.main_grad.add_(tmp_grad)

                tmp_grad._clear_data()

        return param_hook

    def forward(self, *inputs, **kwargs):
        outputs = self._layers(*inputs, **kwargs)

        return outputs

    def state_dict(
        self,
        destination=None,
        include_sublayers=True,
        structured_name_prefix="",
    ):
        return self._layers.state_dict(
            destination=destination,
            include_sublayers=include_sublayers,
            structured_name_prefix=structured_name_prefix,
        )

    @framework.deprecate_stat_dict
    def set_state_dict(self, state_dict, use_structured_name=True):
        self._layers.set_state_dict(
            state_dict, use_structured_name=use_structured_name
        )


class MixPrecisionOptimizer:
    def __init__(self, optimizer):
        self._inner_opt = optimizer
        self._parameter_list = obtain_optimizer_parameters_list(optimizer)

    @imperative_base.no_grad
    @framework.dygraph_only
    def step(self):
        if not isinstance(self._parameter_list[0], dict):
            params_grads = []
            for param in self._parameter_list:
                if param.stop_gradient:
                    continue
                grad_var = param.main_grad
                if grad_var is None:
                    continue
                if paddle.in_dynamic_mode():
                    if (
                        hasattr(grad_var, "is_selected_rows")
                        and grad_var.is_selected_rows()
                        and self._inner_opt.regularization is not None
                    ):
                        raise RuntimeError(
                            "AdamW don't support weight_decay with sparse parameters, please set it to None."
                        )
                else:
                    if (
                        hasattr(grad_var, "_is_sparse")
                        and grad_var._is_sparse()
                        and self._inner_opt.regularization is not None
                    ):
                        raise RuntimeError(
                            "AdamW don't support weight_decay with sparse parameters, please set it to None."
                        )
                params_grads.append((param, grad_var))

            optimize_ops = self._inner_opt._apply_optimize(
                loss=None, startup_program=None, params_grads=params_grads
            )
        else:
            # optimize parameters in groups
            for param_group in self._inner_opt._param_groups:
                params_grads = defaultdict(lambda: [])
                for param in param_group['params']:
                    if param.stop_gradient:
                        continue
                    grad_var = param.main_grad
                    if grad_var is None:
                        continue
                    if paddle.in_dynamic_mode():
                        if (
                            hasattr(grad_var, "is_selected_rows")
                            and grad_var.is_selected_rows()
                            and self._inner_opt.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    else:
                        if (
                            hasattr(grad_var, "_is_sparse")
                            and grad_var._is_sparse()
                            and self._inner_opt.regularization is not None
                        ):
                            raise RuntimeError(
                                "AdamW don't support weight_decay with sparse parameters, please set it to None."
                            )
                    params_grads['params'].append((param, grad_var))
                params_grads.update(
                    {k: v for k, v in param_group.items() if k != 'params'}
                )
                self._apply_optimize(
                    loss=None, startup_program=None, params_grads=params_grads
                )

    @framework.dygraph_only
    def clear_grad(self, set_to_zero=True):
        param_list = []
        if self._parameter_list is None or not isinstance(
            self._parameter_list[0], dict
        ):
            for p in self._parameter_list:
                if not p.stop_gradient:
                    param_list.append(p)
        else:
            for param_group in self._param_groups:
                for p in param_group['params']:
                    if not p.stop_gradient:
                        param_list.append(p)

        for p in param_list:
            if hasattr(p, "main_grad") and p.main_grad is not None:
                if set_to_zero:
                    p.main_grad.zero_()
                else:
                    p.main_grad._clear()
                    p.main_grad = None
            elif not hasattr(p, "main_grad"):
                p.clear_gradient(set_to_zero)

    def __getattr__(self, item):
        return getattr(self._inner_opt, item)


def unscale_method(self, optimizer):
    if not self._enable:
        return
    param_grads = []
    if getattr(optimizer, '_param_groups', None) and isinstance(
        optimizer._param_groups[0], dict
    ):
        for group in optimizer._param_groups:
            for param in group['params']:
                if param.main_grad is not None:
                    assert param.main_grad.dtype == paddle.float32
                    param_grads.append(param.main_grad)
    else:
        for param in optimizer._parameter_list:
            if param.main_grad is not None:
                assert param.main_grad.dtype == paddle.float32
                param_grads.append(param.main_grad)

    temp_found_inf = paddle.to_tensor(np.array([0]).astype(np.bool_))
    if len(param_grads):
        _legacy_C_ops.check_finite_and_unscale(
            param_grads,
            self._scale,
            param_grads,
            temp_found_inf,
        )

    self._found_inf = 1 if temp_found_inf else 0

    hcg = fleet.get_hybrid_communicate_group()
    if hcg is not None and hcg.nranks > hcg.get_data_parallel_world_size():
        is_found_inf = paddle.to_tensor([self._found_inf], dtype="int32")
        paddle.distributed.all_reduce(
            is_found_inf, op=paddle.distributed.ReduceOp.MAX, group=None
        )
        self._found_inf = int(is_found_inf)


@deprecated(
    since="2.5.0",
    update_to="paddle.distributed_scaler",
    level=1,
)
class MixPrecisionScaler:
    def __init__(self, scaler):
        self._inner_scaler = scaler
        self._inner_scaler._unscale = MethodType(unscale_method, scaler)

    def __getattr__(self, item):
        return getattr(self._inner_scaler, item)
