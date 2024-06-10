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
from .assign import (
    Assign,
    NumpyArrayInitializer,  # noqa: F401
)
from .bilinear import Bilinear
from .constant import (
    Constant,
    ConstantInitializer,  # noqa: F401
)
from .dirac import Dirac
from .initializer import (
    Initializer,  # noqa: F401
    calculate_gain,
)
from .kaiming import (
    KaimingNormal,
    KaimingUniform,
    MSRAInitializer,  # noqa: F401
)
from .normal import (
    Normal,
    NormalInitializer,  # noqa: F401
    TruncatedNormal,
    TruncatedNormalInitializer,  # noqa: F401
)
from .orthogonal import Orthogonal
from .uniform import (
    Uniform,
    UniformInitializer,  # noqa: F401
)
from .xavier import (
    XavierInitializer,  # noqa: F401
    XavierNormal,
    XavierUniform,
)

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
