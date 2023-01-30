# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

<<<<<<< HEAD
import paddle
import paddle.autograd as imperative_base
from paddle import _legacy_C_ops

from ...base.topology import ParallelMode
=======
from __future__ import print_function
import sys
from paddle.optimizer import Optimizer
from ...base.topology import ParallelMode
from paddle.fluid.dygraph import base as imperative_base
from paddle.fluid import framework
from paddle.fluid.framework import Variable
import types
from paddle.fluid import core
import paddle
from paddle import _C_ops, _legacy_C_ops
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

__all__ = []


class HybridParallelGradScaler:
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, scaler, hcg):
        self._scaler = scaler
        self._hcg = hcg
        self._use_dp_mode = (
<<<<<<< HEAD
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL
        )
=======
            self._hcg.get_parallel_mode() == ParallelMode.DATA_PARALLEL)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def scale(self, var):
        return self._scaler.scale(var)

    def minimize(self, optimizer, *args, **kwargs):
        if not self._enable:
            return optimizer.minimize(*args, **kwargs)

        #  unscale the grad
        self._unscale(optimizer)

        optimize_ops, params_grads = (None, None)

<<<<<<< HEAD
        optimizer._set_auxiliary_var('found_inf', self._found_inf)
        optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
        self._cache_founf_inf = optimizer._get_auxiliary_var('found_inf')
=======
        if self._found_inf:
            self._cache_founf_inf = True
        else:
            optimize_ops, params_grads = optimizer.minimize(*args, **kwargs)
            self._cache_founf_inf = False
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

        if self._use_dynamic_loss_scaling:
            self._update()

        return optimize_ops, params_grads

<<<<<<< HEAD
    @imperative_base.no_grad()
=======
    @imperative_base.no_grad
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def _unscale(self, optimizer):
        if not self._enable:
            return
        param_grads = [
<<<<<<< HEAD
            param._grad_ivar()
            for param in optimizer._parameter_list
            if param._grad_ivar() is not None
        ]
        _legacy_C_ops.check_finite_and_unscale(
            param_grads, self._scale, param_grads, self._found_inf
        )
=======
            param._grad_ivar() for param in optimizer._parameter_list
            if param._grad_ivar() is not None
        ]
        _legacy_C_ops.check_finite_and_unscale(param_grads, self._scale,
                                               param_grads, self._found_inf)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        # allreduce_max found_inf in check_group
        if not self._use_dp_mode:
            self._found_inf = paddle.cast(self._found_inf, dtype="int32")
            # TODO(shenliang03) Since the minimize call in the optimizer is
            # after the gradscaler, check_finite needs to synchronize global
            # information. In the future, we should use check_group
<<<<<<< HEAD
            paddle.distributed.all_reduce(
                self._found_inf, op=paddle.distributed.ReduceOp.MAX, group=None
            )
=======
            paddle.distributed.all_reduce(self._found_inf,
                                          op=paddle.distributed.ReduceOp.MAX,
                                          group=None)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            self._found_inf = paddle.cast(self._found_inf, dtype="bool")

    def __getattr__(self, item):
        return getattr(self._scaler, item)
