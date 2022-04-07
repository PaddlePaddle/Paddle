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

import paddle
from paddle.distribution import distribution
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode


class ExponentialFamily(distribution.Distribution):
    r""" 
    ExponentialFamily is the base class for probability distributions belonging 
    to exponential family, whose probability mass/density function has the 
    form is defined below

    ExponentialFamily is derived from `paddle.distribution.Distribution`.
    
    .. math::

        f_{F}(x; \theta) = \exp(\langle t(x), \theta\rangle - F(\theta) + k(x))
    
    where :math:`\theta` denotes the natural parameters, :math:`t(x)` denotes 
    the sufficient statistic, :math:`F(\theta)` is the log normalizer function 
    for a given family and :math:`k(x)` is the carrier measure.

    Distribution belongs to exponential family referring to https://en.wikipedia.org/wiki/Exponential_family
    """

    @property
    def _natural_parameters(self):
        raise NotImplementedError

    def _log_normalizer(self):
        raise NotImplementedError

    @property
    def _mean_carrier_measure(self):
        raise NotImplementedError

    def entropy(self):
        """caculate entropy use `bregman divergence` 
        https://www.lix.polytechnique.fr/~nielsen/EntropyEF-ICIP2010.pdf
        """
        entropy_value = -self._mean_carrier_measure

        natural_parameters = []
        for parameter in self._natural_parameters:
            parameter = parameter.detach()
            parameter.stop_gradient = False
            natural_parameters.append(parameter)

        log_norm = self._log_normalizer(*natural_parameters)

        if _non_static_mode():
            grads = paddle.grad(
                log_norm.sum(), natural_parameters, create_graph=True)
        else:
            grads = paddle.static.gradients(log_norm.sum(), natural_parameters)

        entropy_value += log_norm
        for p, g in zip(natural_parameters, grads):
            entropy_value -= p * g

        return entropy_value
