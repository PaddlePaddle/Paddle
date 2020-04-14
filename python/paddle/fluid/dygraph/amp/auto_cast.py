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
from ...wrapped_decorator import signature_safe_contextmanager, wrap_decorator
from paddle.fluid import core
import contextlib
from ...framework import Variable, in_dygraph_mode, OpProtoHolder, Parameter, _dygraph_tracer, dygraph_only
import warnings

__all__ = ['autocast']


@signature_safe_contextmanager
@dygraph_only
def autocast(enable=True):
    if enable and not core.is_compiled_with_cuda():
        warnings.warn(
            'Auto Mixed Precision can only be enabled with Paddle compiled with CUDA.'
        )
        enable = False
    tracer = _dygraph_tracer()
    if tracer:
        original_val = tracer._enable_autocast
        tracer._enable_autocast = enable
    yield
    if tracer:
        tracer._enable_autocast = original_val
