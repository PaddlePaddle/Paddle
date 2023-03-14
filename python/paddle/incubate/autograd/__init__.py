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
from .functional import Hessian, Jacobian, jvp, vjp
from .primapi import forward_grad, grad
from .primx import prim2orig
from .utils import disable_prim, enable_prim, prim_enabled

__all__ = [  # noqa
    'vjp',
    'jvp',
    'Jacobian',
    'Hessian',
    'enable_prim',
    'disable_prim',
    'forward_grad',
    'grad',
]
