# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import sys as _sys

from paddle.distribution import *  # noqa: F403
from paddle.distribution import (
    __all__ as __all__,
    bernoulli,
    beta,
    binomial,
    categorical,
    cauchy,
    chi2,
    constraint,
    continuous_bernoulli,
    dirichlet,
    distribution,
    exponential,
    exponential_family,
    gamma,
    geometric,
    gumbel,
    independent,
    kl,
    laplace,
    lkj_cholesky,
    lognormal,
    multinomial,
    multivariate_normal,
    normal,
    poisson,
    student_t,
    transform,
    transformed_distribution,
    uniform,
    variable,
)

_sys.modules['paddle.distributions.bernoulli'] = bernoulli
_sys.modules['paddle.distributions.beta'] = beta
_sys.modules['paddle.distributions.binomial'] = binomial
_sys.modules['paddle.distributions.categorical'] = categorical
_sys.modules['paddle.distributions.cauchy'] = cauchy
_sys.modules['paddle.distributions.chi2'] = chi2
_sys.modules['paddle.distributions.constraint'] = constraint
_sys.modules['paddle.distributions.continuous_bernoulli'] = continuous_bernoulli
_sys.modules['paddle.distributions.dirichlet'] = dirichlet
_sys.modules['paddle.distributions.distribution'] = distribution
_sys.modules['paddle.distributions.exponential'] = exponential
_sys.modules['paddle.distributions.exponential_family'] = exponential_family
_sys.modules['paddle.distributions.gamma'] = gamma
_sys.modules['paddle.distributions.geometric'] = geometric
_sys.modules['paddle.distributions.gumbel'] = gumbel
_sys.modules['paddle.distributions.independent'] = independent
_sys.modules['paddle.distributions.kl'] = kl
_sys.modules['paddle.distributions.laplace'] = laplace
_sys.modules['paddle.distributions.lkj_cholesky'] = lkj_cholesky
_sys.modules['paddle.distributions.lognormal'] = lognormal
_sys.modules['paddle.distributions.multinomial'] = multinomial
_sys.modules['paddle.distributions.multivariate_normal'] = multivariate_normal
_sys.modules['paddle.distributions.normal'] = normal
_sys.modules['paddle.distributions.poisson'] = poisson
_sys.modules['paddle.distributions.student_t'] = student_t
_sys.modules['paddle.distributions.transform'] = transform
_sys.modules['paddle.distributions.transformed_distribution'] = (
    transformed_distribution
)
_sys.modules['paddle.distributions.uniform'] = uniform
_sys.modules['paddle.distributions.variable'] = variable
