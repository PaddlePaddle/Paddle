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

from .tensor.compat_softmax import log_softmax, softmax
from .tensor.creation import assign
from .tensor.math import (
    erf as _erf,
    expm1,
    i0,
    i0e,
    i1,
    i1e,
    log1p,
    logit,
    logsumexp,
    sinc as _sinc,
)
from .utils.decorator_utils import param_one_alias

__all__ = [
    "erf",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "log1p",
    "log_softmax",
    "logit",
    "logsumexp",
    "sinc",
    "softmax",
    "expm1",
]


@param_one_alias(["x", "input"])
def erf(x, name=None, *, out=None):
    result = _erf(x, name=name)
    return assign(result, out) if out is not None else result


@param_one_alias(["x", "input"])
def sinc(x, name=None, *, out=None):
    result = _sinc(x, name=name)
    return assign(result, out) if out is not None else result
