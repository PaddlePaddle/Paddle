# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
from functools import wraps

import paddle


# NOTE(zhengtianyu): align ClipGradByGlobalNorm in auto_parallel_align_mode.
# In old dygraph semi-auto parallel, each rank has parameter and gradient information
# from other ranks. To align with this behavior, this decorator ensures auto_hybrid_pp
# uses the same logic as old dygraph semi-auto parallel for ClipGradByGlobalNorm in align mode.
# Pay attention to the auto_hybrid_pp's default logic matches dynamic manual-parallel,
# Refer to NOTE: Fix grad_clip in auto_hybrid_pp mode
def _patch_grads_for_step(
    amp_master_grad=False,
):
    """
    Only for auto parallel align mode, use this decorator to handle None gradients in optimizer step.

    This decorator is applied to optimizer step methods to handle cases where parameters
    have None gradients. It creates zero gradients for parameters that need gradients
    but currently have None gradients.

    Args:
        amp_master_grad (bool, optional): Whether to use master gradient mode.
            If True, gradients will be created as float32 regardless of parameter dtype.
            If False, gradients will be created with the same dtype as the parameter.
            Default is False.

    Returns:
        function: Decorated step method that handles None gradients.

    Example:
        .. code-block:: python

            >>> from __future__ import annotations
            >>> import paddle.distributed as dist
            >>> import types
            >>> from paddle.distributed.auto_parallel._utils import _patch_grads_for_step

            >>> opt = paddle.optimizer.AdamW(
            ...     learning_rate=0.001,
            ...     parameters=self.model.parameters(),
            ...     grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
            ...     )
            >>> if dist.in_auto_parallel_align_mode():
            >>>     orig_step = (
            ...         opt.step.__func__ if hasattr(opt.step, "__func__") else opt.step
            ...     )
            >>>     decorator = (
            ...         _patch_grads_for_step(
            ...             amp_master_grad=True
            ...         )
            ...     )
            >>>     new_step = decorator(orig_step)
            >>>     opt.step = types.MethodType(new_step, opt)

    """

    def decorator(step_method):
        @wraps(step_method)
        def wrapper(self, *args, **kwargs):
            # Helper function to set gradient for a parameter
            def set_param_grad(param):
                if param.stop_gradient or param.grad is not None:
                    return

                if hasattr(param, "main_grad"):
                    param.main_grad = paddle.zeros_like(
                        param, dtype=paddle.float32
                    )
                else:
                    dtype = paddle.float32 if amp_master_grad else param.dtype
                    param.grad = paddle.zeros_like(param, dtype=dtype)

            if not isinstance(self._parameter_list[0], dict):
                for param in self._parameter_list:
                    set_param_grad(param)
            else:
                for param_group in self._param_groups:
                    for param in param_group['params']:
                        set_param_grad(param)
            return step_method(self, *args, **kwargs)

        return wrapper

    return decorator
