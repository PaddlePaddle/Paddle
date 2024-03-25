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


import numpy as np

import paddle
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import exponential_family
from paddle.framework import in_dynamic_mode
from paddle.nn.functional import (
    binary_cross_entropy_with_logits,
    sigmoid,
    softplus,
)

# Smallest representable number
EPS = {
    'float32': paddle.finfo(paddle.float32).eps,
    'float64': paddle.finfo(paddle.float64).eps,
}


def _clip_probs(probs, dtype):
    """Clip probs from [0, 1] to (0, 1) with ``eps``.

    Args:
        probs (Tensor): probs of Bernoulli.
        dtype (str): data type.

    Returns:
        Tensor: Clipped probs.
    """
    eps = EPS.get(dtype)
    return paddle.clip(probs, min=eps, max=1 - eps).astype(dtype)


class Bernoulli(exponential_family.ExponentialFamily):
    r"""Bernoulli distribution parameterized by ``probs``, which is the probability of value 1.

    In probability theory and statistics, the Bernoulli distribution, named after Swiss
    mathematician Jacob Bernoulli, is the discrete probability distribution of a random
    variable which takes the value 1 with probability ``p`` and the value 0 with
    probability ``q=1-p``.

    The probability mass function of this distribution, over possible outcomes ``k``, is

    .. math::

        {\begin{cases}
        q=1-p & \text{if }value=0 \\
        p & \text{if }value=1
        \end{cases}}

    Args:
        probs (float|Tensor): The ``probs`` input of Bernoulli distribution. The data type is float32 or float64. The range must be in [0, 1].
        name (str, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:

        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Bernoulli

            >>> # init `probs` with a float
            >>> rv = Bernoulli(probs=0.3)

            >>> print(rv.mean)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.30000001)

            >>> print(rv.variance)
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.21000001)

            >>> print(rv.entropy())
            Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
            0.61086434)
    """

    def __init__(self, probs, name=None):
        self.name = name or 'Bernoulli'
        if not in_dynamic_mode():
            check_type(
                probs,
                'probs',
                (float, Variable),
                self.name,
            )

        # Get/convert probs to tensor.
        if self._validate_args(probs):
            self.probs = probs
            self.dtype = convert_dtype(probs.dtype)
        else:
            [self.probs] = self._to_tensor(probs)
            self.dtype = paddle.get_default_dtype()

        # Check probs range [0, 1].
        if in_dynamic_mode():
            """Not use `paddle.any` in static mode, which always be `True`."""
            if (
                paddle.any(self.probs < 0)
                or paddle.any(self.probs > 1)
                or paddle.any(paddle.isnan(self.probs))
            ):
                raise ValueError("The arg of `probs` must be in range [0, 1].")

        # Clip probs from [0, 1] to (0, 1) with smallest representable number `eps`.
        self.probs = _clip_probs(self.probs, self.dtype)
        self.logits = self._probs_to_logits(self.probs, is_binary=True)

        super().__init__(batch_shape=self.probs.shape, event_shape=())

    @property
    def mean(self):
        """Mean of Bernoulli distribution.

        Returns:
            Tensor: Mean value of distribution.
        """
        return self.probs

    @property
    def variance(self):
        """Variance of Bernoulli distribution.

        Returns:
            Tensor: Variance value of distribution.
        """
        return paddle.multiply(self.probs, (1 - self.probs))

    def sample(self, shape):
        """Sample from Bernoulli distribution.

        Args:
            shape (Sequence[int]): Sample shape.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(paddle.full((1), 0.3))
                >>> print(rv.sample([100]).shape)
                [100, 1]

                >>> rv = Bernoulli(paddle.to_tensor(0.3))
                >>> print(rv.sample([100]).shape)
                [100]

                >>> rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))
                >>> print(rv.sample([100]).shape)
                [100, 2]

                >>> rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))
                >>> print(rv.sample([100, 2]).shape)
                [100, 2, 2]
        """
        name = self.name + '_sample'
        if not in_dynamic_mode():
            check_type(
                shape,
                'shape',
                (np.ndarray, Variable, list, tuple),
                name,
            )

        shape = shape if isinstance(shape, tuple) else tuple(shape)
        shape = self._extend_shape(shape)

        with paddle.no_grad():
            return paddle.bernoulli(self.probs.expand(shape), name=name)

    def rsample(self, shape, temperature=1.0):
        """Sample from Bernoulli distribution (reparameterized).

        The `rsample` is a continuously approximate of Bernoulli distribution reparameterized sample method.
        [1] Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables. 2016.
        [2] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. 2016.

        Note:
            `rsample` need to be followed by a `sigmoid`, which converts samples' value to unit interval (0, 1).

        Args:
            shape (Sequence[int]): Sample shape.
            temperature (float): temperature for rsample, must be positive.

        Returns:
            Tensor: Sampled data with shape `sample_shape` + `batch_shape` + `event_shape`.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> paddle.seed(1)
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(paddle.full((1), 0.3))
                >>> print(rv.sample([100]).shape)
                [100, 1]

                >>> rv = Bernoulli(0.3)
                >>> print(rv.rsample([100]).shape)
                [100]

                >>> rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))
                >>> print(rv.rsample([100]).shape)
                [100, 2]

                >>> rv = Bernoulli(paddle.to_tensor([0.3, 0.5]))
                >>> print(rv.rsample([100, 2]).shape)
                [100, 2, 2]

                >>> # `rsample` has to be followed by a `sigmoid`
                >>> rv = Bernoulli(0.3)
                >>> rsample = rv.rsample([3, ])
                >>> rsample_sigmoid = paddle.nn.functional.sigmoid(rsample)
                >>> print(rsample)
                Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-1.46112013, -0.01239836, -1.32765460])
                >>> print(rsample_sigmoid)
                Tensor(shape=[3], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.18829606, 0.49690047, 0.20954758])

                >>> # The smaller the `temperature`, the distribution of `rsample` closer to `sample`, with `probs` of 0.3.
                >>> print(paddle.nn.functional.sigmoid(rv.rsample([1000, ], temperature=1.0)).sum())
                >>> # doctest: +SKIP('output will be different')
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                365.63122559)
                >>> # doctest: -SKIP

                >>> print(paddle.nn.functional.sigmoid(rv.rsample([1000, ], temperature=0.1)).sum())
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                320.15057373)
        """
        name = self.name + '_rsample'
        if not in_dynamic_mode():
            check_type(
                shape,
                'shape',
                (np.ndarray, Variable, list, tuple),
                name,
            )
            check_type(
                temperature,
                'temperature',
                (float,),
                name,
            )

        shape = shape if isinstance(shape, tuple) else tuple(shape)
        shape = self._extend_shape(shape)

        temperature = paddle.full(
            shape=(), fill_value=temperature, dtype=self.dtype
        )

        probs = self.probs.expand(shape)
        uniforms = paddle.rand(shape, dtype=self.dtype)
        return paddle.divide(
            paddle.add(
                paddle.subtract(uniforms.log(), (-uniforms).log1p()),
                paddle.subtract(probs.log(), (-probs).log1p()),
            ),
            temperature,
        )

    def cdf(self, value):
        r"""Cumulative distribution function(CDF) evaluated at value.

        .. math::

            { \begin{cases}
            0 & \text{if } value \lt  0 \\
            1 - p & \text{if } 0 \leq value \lt  1 \\
            1 & \text{if } value \geq 1
            \end{cases}
            }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: CDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(0.3)
                >>> print(rv.cdf(paddle.to_tensor([1.0])))
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [1.])
        """
        name = self.name + '_cdf'
        if not in_dynamic_mode():
            check_type(value, 'value', Variable, name)

        value = self._check_values_dtype_in_probs(self.probs, value)
        probs, value = paddle.broadcast_tensors([self.probs, value])

        zeros = paddle.zeros_like(probs)
        ones = paddle.ones_like(probs)

        return paddle.where(
            value < 0,
            zeros,
            paddle.where(value < 1, paddle.subtract(ones, probs), ones),
            name=name,
        )

    def log_prob(self, value):
        """Log of probability density function.

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: Log of probability density evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(0.3)
                >>> print(rv.log_prob(paddle.to_tensor([1.0])))
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-1.20397282])
        """
        name = self.name + '_log_prob'
        if not in_dynamic_mode():
            check_type(value, 'value', Variable, name)

        value = self._check_values_dtype_in_probs(self.probs, value)
        logits, value = paddle.broadcast_tensors([self.logits, value])
        return -binary_cross_entropy_with_logits(
            logits, value, reduction='none', name=name
        )

    def prob(self, value):
        r"""Probability density function(PDF) evaluated at value.

        .. math::

            { \begin{cases}
                q=1-p & \text{if }value=0 \\
                p & \text{if }value=1
                \end{cases}
            }

        Args:
            value (Tensor): Value to be evaluated.

        Returns:
            Tensor: PDF evaluated at value.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(0.3)
                >>> print(rv.prob(paddle.to_tensor([1.0])))
                Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.29999998])
        """
        name = self.name + '_prob'
        if not in_dynamic_mode():
            check_type(value, 'value', Variable, name)

        return self.log_prob(value).exp(name=name)

    def entropy(self):
        r"""Entropy of Bernoulli distribution.

        .. math::

            {
                entropy = -(q \log q + p \log p)
            }

        Returns:
            Tensor: Entropy of distribution.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(0.3)
                >>> print(rv.entropy())
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                0.61086434)
        """
        name = self.name + '_entropy'

        return binary_cross_entropy_with_logits(
            self.logits, self.probs, reduction='none', name=name
        )

    def kl_divergence(self, other):
        r"""The KL-divergence between two Bernoulli distributions.

        .. math::

            {
                KL(a || b) = p_a \log(p_a / p_b) + (1 - p_a) \log((1 - p_a) / (1 - p_b))
            }

        Args:
            other (Bernoulli): instance of Bernoulli.

        Returns:
            Tensor: kl-divergence between two Bernoulli distributions.

        Examples:

            .. code-block:: python

                >>> import paddle
                >>> from paddle.distribution import Bernoulli

                >>> rv = Bernoulli(0.3)
                >>> rv_other = Bernoulli(0.7)

                >>> print(rv.kl_divergence(rv_other))
                Tensor(shape=[], dtype=float32, place=Place(cpu), stop_gradient=True,
                0.33891910)
        """
        name = self.name + '_kl_divergence'
        if not in_dynamic_mode():
            check_type(other, 'other', Bernoulli, name)

        a_logits = self.logits
        b_logits = other.logits

        log_pa = -softplus(-a_logits)
        log_pb = -softplus(-b_logits)

        pa = sigmoid(a_logits)
        one_minus_pa = sigmoid(-a_logits)

        log_one_minus_pa = -softplus(a_logits)
        log_one_minus_pb = -softplus(b_logits)

        return paddle.add(
            paddle.subtract(
                paddle.multiply(log_pa, pa), paddle.multiply(log_pb, pa)
            ),
            paddle.subtract(
                paddle.multiply(log_one_minus_pa, one_minus_pa),
                paddle.multiply(log_one_minus_pb, one_minus_pa),
            ),
        )
