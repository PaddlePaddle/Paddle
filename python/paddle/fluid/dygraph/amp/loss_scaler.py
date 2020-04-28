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
from paddle.fluid import core, layers, dygraph
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
import warnings

__all__ = ['Scaler']


class Scaler(object):
    def __init__(self,
                 init_scale=2.**16,
                 growth_factor=2.0,
                 backoff_factor=0.5,
                 growth_interval=2000,
                 enable=True):
        if enable and not core.is_compiled_with_cuda():
            warnings.warn(
                'Auto Mixed Precision can only be enabled with Paddle compiled with CUDA.'
            )
            self._enable = False
        else:
            self._enable = enable

        if self._enable:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

            self._init_scale = init_scale
            # self._scale will be lazily initialized during the first call to scale()
            self._scale = init_scale
            self._growth_factor = growth_factor
            self._backoff_factor = backoff_factor
            self._growth_interval = growth_interval
            #self._init_growth_tracker = 0
            # self._growth_tracker will be lazily initialized during the first call to scale()
            self._growth_tracker = 0
            # READY = self.READY
            # self._per_optimizer_states = defaultdict(lambda: {"stage": READY, "found_inf_per_device": {}})
            self._found_inf = False

    def scale(self, output):
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`Scaler` is not enabled, outputs are returned
        unmodified.

        Arguments:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        if not self._enable:
            return output

        if isinstance(output, core.VarBase):
            return output * self._scale
        elif isinstance(outputs, list) or isinstance(outputs, tuple):
            return [self.scale(var) for var in output]
        else:
            raise ValueError(
                "output must be a Variable or an list/tuple of Variables")

    @dygraph.no_grad
    def scale_inplace(self, var, scale):
        #TODO(zhiqiu): support inplace operation directly in the future
        new_var = core.ops.scale(var, 'scale', scale)
        layers.assign(new_var, var)
        return None

    def unscale_(self, optimizer):
        if not self._enable:
            return
        inv_scale = 1.0 / self._scale
        for param in optimizer._parameter_list:
            if param._grad_ivar() is not None:
                if layers.isfinite(param._grad_ivar()):
                    self.scale_inplace(param._grad_ivar(), inv_scale)
                else:
                    #print(param._grad_ivar())
                    self._found_inf = True
                    return
        self._found_inf = False

    def step(self, optimizer, *args, **kwargs):
        if not self._enable:
            return

        #  unscale the grad
        self.unscale_(optimizer)

        if not self._found_inf:
            optimizer.minimize(*args, **kwargs)

    def update(self, new_scale=None):
        """
        Updates the scale factor.
        """
        if not self._enable:
            return

        if new_scale is not None:
            self._scale = new_scale
            return

        if self._found_inf:
            self._scale = float(self._scale) * self._backoff_factor
            self._growth_tracker = 0
            print('found infinite', self._scale, self._backoff_factor)
        else:
            self._growth_tracker = self._growth_tracker + 1
            if self._growth_tracker == self._growth_interval:
                self._scale = self._scale * self._growth_factor
                self._growth_tracker = 0

        return
