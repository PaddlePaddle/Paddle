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

from .beta import Beta
from .categorical import Categorical
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exponential_family import ExponentialFamily
from .kl import kl_divergence, register_kl
from .multinomial import Multinomial
from .normal import Normal
from .uniform import Uniform

__all__ = [  # noqa
    'Beta',
    'Categorical',
    'Dirichlet',
    'Distribution',
    'ExponentialFamily',
    'Multinomial',
    'Normal',
    'Uniform',
    'kl_divergence',
    'register_kl',
]
