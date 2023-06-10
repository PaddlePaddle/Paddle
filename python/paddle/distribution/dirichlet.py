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
from paddle.distribution import exponential_family
from paddle.fluid.data_feeder import check_variable_and_dtype
from paddle.fluid.layer_helper import LayerHelper
from paddle.framework import in_dynamic_mode


class Dirichlet(exponential_family.ExponentialFamily):
    r"""
    Dirichlet distribution with parameter "concentration".

    The Dirichlet distribution is defined over the `(k-1)-simplex` using a
    positive, lenght-k vector concentration(`k > 1`).
    The Dirichlet is identically the Beta distribution when `k = 2`.

    For independent and identically distributed continuous random variable
    :math:`\boldsymbol X \in R_k` , and support
    :math:`\boldsymbol X \in (0,1), ||\boldsymbol X|| = 1` ,
    The probability density function (pdf) is

    .. math::

        f(\boldsymbol X; \boldsymbol \alpha) = \frac{1}{B(\boldsymbol \alpha)} \prod_{i=1}^{k}x_i^{\alpha_i-1}

    where :math:`\boldsymbol \alpha = {\alpha_1,...,\alpha_k}, k \ge 2` is
    parameter, the normalizing constant is the multivariate beta function.

    .. math::

        B(\boldsymbol \alpha) = \frac{\prod_{i=1}^{k} \Gamma(\alpha_i)}{\Gamma(\alpha_0)}

    :math:`\alpha_0=\sum_{i=1}^{k} \alpha_i` is the sum of parameters,
    :math:`\Gamma(\alpha)` is gamma function.

    Args:
        concentration (Tensor): "Concentration" parameter of dirichlet
            distribution, also called :math:`\alpha`. When it's over one
            dimension, the last axis denotes the parameter of distribution,
            ``event_shape=concentration.shape[-1:]`` , axes other than last are
            condsider batch dimensions with ``batch_shape=concentration.shape[:-1]`` .

    Examples:

        .. code-block:: python

            import paddle

            dirichlet = paddle.distribution.Dirichlet(paddle.to_tensor([1., 2., 3.]))

            print(dirichlet.entropy())
            # Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        -1.24434423)
            print(dirichlet.prob(paddle.to_tensor([.3, .5, .6])))
            # Tensor(shape=[], dtype=float32, place=CUDAPlace(0), stop_gradient=True,
            #        10.80000114)

    """

    def __init__(self, concentration):
        if concentration.dim() < 1:
            raise ValueError(
                "`concentration` parameter must be at least one dimensional"
            )

        self.concentration = concentration
        super().__init__(concentration.shape[:-1], concentration.shape[-1:])

    @property
    def mean(self):
        """Mean of Dirichelt distribution.

        Returns:
            Mean value of distribution.
        """
        return self.concentration / self.concentration.sum(-1, keepdim=True)

    @property
    def variance(self):
        """Variance of Dirichlet distribution.

        Returns:
            Variance value of distribution.
        """
        concentration0 = self.concentration.sum(-1, keepdim=True)
        return (self.concentration * (concentration0 - self.concentration)) / (
            concentration0.pow(2) * (concentration0 + 1)
        )

    def sample(self, shape=()):
        """Sample from dirichlet distribution.

        Args:
            shape (Sequence[int], optional): Sample shape. Defaults to empty tuple.
        """
        shape = shape if isinstance(shape, tuple) else tuple(shape)
        return _dirichlet(self.concentration.expand(self._extend_shape(shape)))

    def prob(self, value):
        """Probability density function(PDF) evaluated at value.

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            PDF evaluated at value.
        """
        return paddle.exp(self.log_prob(value))

    def log_prob(self, value):
        """Log of probability densitiy function.

        Args:
            value (Tensor): Value to be evaluated.
        """
        return (
            (paddle.log(value) * (self.concentration - 1.0)).sum(-1)
            + paddle.lgamma(self.concentration.sum(-1))
            - paddle.lgamma(self.concentration).sum(-1)
        )

    def entropy(self):
        """Entropy of Dirichlet distribution.

        Returns:
            Entropy of distribution.
        """
        concentration0 = self.concentration.sum(-1)
        k = self.concentration.shape[-1]
        return (
            paddle.lgamma(self.concentration).sum(-1)
            - paddle.lgamma(concentration0)
            - (k - concentration0) * paddle.digamma(concentration0)
            - (
                (self.concentration - 1.0) * paddle.digamma(self.concentration)
            ).sum(-1)
        )

    @property
    def _natural_parameters(self):
        return (self.concentration,)

    def _log_normalizer(self, x):
        return x.lgamma().sum(-1) - paddle.lgamma(x.sum(-1))


def _dirichlet(concentration, name=None):

    if in_dynamic_mode():
        return paddle._C_ops.dirichlet(concentration)
    else:
        op_type = 'dirichlet'
        check_variable_and_dtype(
            concentration,
            'concentration',
            ['float16', 'float32', 'float64', 'uint16'],
            op_type,
        )
        helper = LayerHelper(op_type, **locals())
        out = helper.create_variable_for_type_inference(
            dtype=concentration.dtype
        )
        helper.append_op(
            type=op_type,
            inputs={"Alpha": concentration},
            outputs={'Out': out},
            attrs={},
        )
        return out
