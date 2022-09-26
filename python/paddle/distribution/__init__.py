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

from paddle.distribution import transform
from paddle.distribution.beta import Beta
from paddle.distribution.categorical import Categorical
from paddle.distribution.dirichlet import Dirichlet
from paddle.distribution.distribution import Distribution
from paddle.distribution.exponential_family import ExponentialFamily
from paddle.distribution.independent import Independent
from paddle.distribution.kl import kl_divergence, register_kl
from paddle.distribution.multinomial import Multinomial
from paddle.distribution.normal import Normal
from paddle.distribution.transform import *  # noqa: F403
from paddle.distribution.transformed_distribution import \
    TransformedDistribution
from paddle.distribution.uniform import Uniform
from paddle.distribution.laplace import Laplace

__all__ = [  # noqa
    'Beta', 'Categorical', 'Dirichlet', 'Distribution', 'ExponentialFamily',
    'Multinomial', 'Normal', 'Uniform', 'kl_divergence', 'register_kl',
    'Independent', 'TransformedDistribution', 'Laplace'
]

__all__.extend(transform.__all__)
