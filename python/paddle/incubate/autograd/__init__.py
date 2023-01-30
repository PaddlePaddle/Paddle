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
<<<<<<< HEAD
from .primapi import forward_grad, grad, to_prim
=======
from .primapi import forward_grad, grad
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
from .primx import prim2orig
from .utils import disable_prim, enable_prim, prim_enabled

__all__ = [  # noqa
<<<<<<< HEAD
    'vjp',
    'jvp',
    'Jacobian',
    'Hessian',
    'enable_prim',
    'disable_prim',
    'forward_grad',
    'grad',
    'to_prim',
=======
    'vjp', 'jvp', 'Jacobian', 'Hessian', 'enable_prim', 'disable_prim',
    'forward_grad', 'grad'
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
]
