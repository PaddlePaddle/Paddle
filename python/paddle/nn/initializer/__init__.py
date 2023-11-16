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

# TODO: define the initializers to create a Parameter in neural network
from ...base.initializer import set_global_initializer
from .assign import NumpyArrayInitializer  # noqa: F401
from .assign import Assign
from .Bilinear import Bilinear
from .constant import ConstantInitializer  # noqa: F401
from .constant import Constant
from .dirac import Dirac
from .initializer import Initializer, calculate_gain  # noqa: F401
from .kaiming import MSRAInitializer  # noqa: F401
from .kaiming import KaimingNormal, KaimingUniform
from .normal import (  # noqa: F401
    Normal,
    NormalInitializer,
    TruncatedNormal,
    TruncatedNormalInitializer,
)
from .orthogonal import Orthogonal
from .uniform import Uniform, UniformInitializer  # noqa: F401
from .xavier import XavierInitializer  # noqa: F401
from .xavier import XavierNormal, XavierUniform

__all__ = [
    'Bilinear',
    'Constant',
    'KaimingUniform',
    'KaimingNormal',
    'XavierNormal',
    'XavierUniform',
    'Assign',
    'Normal',
    'TruncatedNormal',
    'Uniform',
    'Orthogonal',
    'Dirac',
    'set_global_initializer',
    'calculate_gain',
]
