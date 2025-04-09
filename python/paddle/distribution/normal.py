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
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import paddle
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution import distribution
from paddle.framework import in_dynamic_mode
from paddle.tensor import random

if TYPE_CHECKING:
    from typing import Union

    from typing_extensions import TypeAlias

    from paddle import Tensor, dtype
    from paddle._typing import NestedSequence

    _NormalLocBase: TypeAlias = Union[float, complex]
    _NormalLocNDArray: TypeAlias = Union[
        np.float32, np.float64, np.complex64, np.complex128
    ]
    _NormalLoc: TypeAlias = Union[
        _NormalLocBase,
        Sequence[_NormalLocBase],
        NestedSequence[_NormalLocBase],
        npt.NDArray[_NormalLocNDArray],
        Tensor,
    ]
    _NormalScale: TypeAlias = Union[
        float,
        Sequence[float],
        NestedSequence[float],
        npt.NDArray[Union[np.float32, np.float64]],
        Tensor,
    ]


class Normal(distribution.Distribution):
    r"""The Normal distribution with location `loc` and `scale` parameters.

    Mathematical details

    If 'loc' is real number, the probability density function (pdf) is

    .. math::

        pdf(x; \mu, \sigma) = \frac{1}{Z}e^{\frac {-0.5 (x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = (2 \pi \sigma^2)^{0.5}

    If 'loc' is complex number, the probability density function (pdf) is

    .. math::

        pdf(x; \mu, \sigma) = \frac{1}{Z}e^{\frac {-(x - \mu)^2}  {\sigma^2} }

    .. math::

        Z = \pi \sigma^2

    In the above equations:

    * :math:`loc = \mu`: is the mean.
    * :math:`scale = \sigma`: is the std.
    * :math:`Z`: is the normalization constant.

    Args:
        loc(int|float|complex|list|tuple|numpy.ndarray|Tensor): The mean of normal distribution.The data type is float32, float64, complex64 and complex128.
        scale(int|float|list|tuple|numpy.ndarray|Tensor): The std of normal distribution.The data type is float32 and float64.
        name(str|None, optional): Name for the operation (optional, default is None). For more information, please refer to :ref:`api_guide_Name`.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> from paddle.distribution import Normal

            >>> # Define a single scalar Normal distribution.
            >>> dist = Normal(loc=0., scale=3.)
            >>> # Define a batch of two scalar valued Normals.
            >>> # The first has mean 1 and standard deviation 11, the second 2 and 22.
            >>> dist = Normal(loc=[1., 2.], scale=[11., 22.])
            >>> # Get 3 samples, returning a 3 x 2 tensor.
            >>> dist.sample([3])

            >>> # Define a batch of two scalar valued Normals.
            >>> # Both have mean 1, but different standard deviations.
            >>> dist = Normal(loc=1., scale=[11., 22.])

            >>> # Complete example
            >>> value_tensor = paddle.to_tensor([0.8], dtype="float32")

            >>> normal_a = Normal([0.], [1.])
            >>> normal_b = Normal([0.5], [2.])
            >>> sample = normal_a.sample([2])
            >>> # a random tensor created by normal distribution with shape: [2, 1]
            >>> entropy = normal_a.entropy()
            >>> print(entropy)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [1.41893852])
            >>> lp = normal_a.log_prob(value_tensor)
            >>> print(lp)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [-1.23893857])
            >>> p = normal_a.probs(value_tensor)
            >>> print(p)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.28969154])
            >>> kl = normal_a.kl_divergence(normal_b)
            >>> print(kl)
            Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
                [0.34939718])
    """

    loc: Tensor
    scale: Tensor
    name: str
    dtype: dtype

    def __init__(
        self, loc: _NormalLoc, scale: _NormalScale, name: str | None = None
    ) -> None:
        if not in_dynamic_mode():
            check_type(
                loc,
                'loc',
                (
                    int,
                    float,
                    complex,
                    np.ndarray,
                    Variable,
                    paddle.pir.Value,
                    list,
                    tuple,
                ),
                'Normal',
            )
            check_type(
                scale,
                'scale',
                (
                    int,
                    float,
                    np.ndarray,
                    Variable,
                    paddle.pir.Value,
                    list,
                    tuple,
                ),
                'Normal',
            )

        self.all_arg_is_float = False
        self.name = name if name is not None else 'Normal'
        self.dtype = 'float32'
        self._complex_gaussian = False

        if isinstance(loc, int):
            loc = float(loc)
        if isinstance(scale, int):
            scale = float(scale)

        if isinstance(loc, (tuple, list)):
            loc = np.array(loc)
            if loc.dtype == np.float64:
                loc = loc.astype('float32')
            if loc.dtype == np.complex128:
                loc = loc.astype('complex64')

        if isinstance(scale, (tuple, list)):
            scale = np.array(scale, dtype=np.float32)

        if (
            isinstance(loc, complex)
            or (
                isinstance(loc, np.ndarray)
                and loc.dtype in [np.complex64, np.complex128]
            )
            or (self._validate_args(loc) and loc.is_complex())
        ):
            self._complex_gaussian = True
            if isinstance(loc, complex) and isinstance(scale, float):
                self.all_arg_is_float = True

            if isinstance(loc, np.ndarray):
                real_dtype = (
                    'float32' if loc.dtype == np.complex64 else 'float64'
                )
                imag_dtype = (
                    'float32' if loc.dtype == np.complex64 else 'float64'
                )
                real = paddle.to_tensor(loc.real, real_dtype)
                imag = paddle.to_tensor(loc.imag, imag_dtype)
                self.loc = paddle.complex(real, imag)
            elif isinstance(loc, complex):
                real = paddle.to_tensor(loc.real, dtype='float32')
                imag = paddle.to_tensor(loc.imag, dtype='float32')
                self.loc = paddle.complex(real, imag)
            else:
                self.loc = loc

            if isinstance(scale, np.ndarray):
                self.scale = paddle.to_tensor(scale, dtype=scale.dtype)
            elif isinstance(scale, float):
                self.scale = paddle.to_tensor(scale, dtype='float32')
            else:
                self.scale = scale

            self.dtype = convert_dtype(self.loc.dtype)
        else:
            if self._validate_args(loc, scale):
                self.loc = loc
                self.scale = scale
                self.dtype = convert_dtype(loc.dtype)
            else:
                if isinstance(loc, float) and isinstance(scale, float):
                    self.all_arg_is_float = True
                if isinstance(loc, np.ndarray) and str(loc.dtype) in [
                    'float32',
                    'float64',
                ]:
                    self.dtype = loc.dtype
                elif isinstance(scale, np.ndarray) and str(scale.dtype) in [
                    'float32',
                    'float64',
                ]:
                    self.dtype = scale.dtype
                self.loc, self.scale = self._to_tensor(loc, scale)
                if self.dtype != convert_dtype(self.loc.dtype):
                    self.loc = paddle.cast(self.loc, dtype=self.dtype)
                    self.scale = paddle.cast(self.scale, dtype=self.dtype)
        super().__init__(self.loc.shape)

    @property
    def mean(self) -> Tensor:
        """Mean of normal distribution.

        Returns:
            Tensor: mean value.
        """
        return self.loc

    @property
    def variance(self) -> Tensor:
        """Variance of normal distribution.

        Returns:
            Tensor: variance value.
        """
        return self.scale.pow(2)

    def sample(self, shape: Sequence[int] = [], seed: int = 0) -> Tensor:
        """Generate samples of the specified shape.

        Args:
            shape (Sequence[int], optional): Shape of the generated samples.
            seed (int): Python integer number.

        Returns:
            Tensor, A tensor with prepended dimensions shape.The data type is float32.

        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        if not in_dynamic_mode():
            check_type(seed, 'seed', (int), 'sample')

        shape = list(shape)
        batch_shape = list((self.loc + self.scale).shape)
        name = self.name + '_sample'
        if -1 in batch_shape:
            output_shape = shape + batch_shape
            fill_shape = list(batch_shape + shape)
            fill_shape[0] = paddle.shape(self.loc + self.scale)[0].item()
            zero_tmp = paddle.full(fill_shape, 0.0, self.dtype)
            zero_tmp_reshape = paddle.reshape(zero_tmp, output_shape)

            zero_tmp_shape = paddle.shape(zero_tmp_reshape)
            normal_random_tmp = random.gaussian(
                zero_tmp_shape,
                mean=(0.0 + 0.0j) if self._complex_gaussian else 0.0,
                std=1.0,
                seed=seed,
                dtype=self.dtype,
            )
            output = normal_random_tmp * (zero_tmp_reshape + self.scale)
            output = paddle.add(output, self.loc, name=name)
            return output
        else:
            output_shape = shape + batch_shape
            output = random.gaussian(
                output_shape,
                mean=(0.0 + 0.0j) if self._complex_gaussian else 0.0,
                std=1.0,
                seed=seed,
                dtype=self.dtype,
            ) * (paddle.zeros(output_shape, dtype=self.dtype) + self.scale)
            output = paddle.add(output, self.loc, name=name)
            if self.all_arg_is_float:
                return paddle.reshape(output, shape, name=name)
            else:
                return output

    def rsample(self, shape: Sequence[int] = []) -> Tensor:
        """Generate reparameterized samples of the specified shape.

        Args:
          shape (Sequence[int], optional): Shape of the generated samples.

        Returns:
          Tensor: A tensor with prepended dimensions shape.The data type is float32.

        """
        if not isinstance(shape, Iterable):
            raise TypeError('sample shape must be Iterable object.')

        shape = self._extend_shape(tuple(shape))
        eps = paddle.normal(
            mean=(0.0 + 0.0j) if self._complex_gaussian else 0.0, shape=shape
        )
        return self.loc + eps * self.scale

    def entropy(self) -> Tensor:
        r"""Shannon entropy in nats.

        If non-complex, the entropy is

        .. math::

            entropy(\sigma) = 0.5 \log (2 \pi e \sigma^2)

        If complex gaussian, the entropy is

        .. math::

            entropy(\sigma) = \log (\pi e \sigma^2) + 1

        In the above equation:

        * :math:`scale = \sigma`: is the std.

        Returns:
            Tensor, Shannon entropy of normal distribution.The data type is float32.

        """
        name = self.name + '_entropy'
        batch_shape = list((self.loc + self.scale).shape)

        if self._complex_gaussian:
            if -1 in batch_shape:
                fill_shape = list(batch_shape)
                fill_shape[0] = paddle.shape(self.loc + self.scale)[0].item()
                fill_dtype = self.scale.dtype
                zero_tmp = paddle.full(fill_shape, 0.0, fill_dtype)
            else:
                zero_tmp = paddle.full(batch_shape, 0.0, self.scale.dtype)
            return paddle.add(
                1.0 + zero_tmp,
                math.log(math.pi) + 2.0 * paddle.log(self.scale + zero_tmp),
                name=name,
            )
        else:
            if -1 in batch_shape:
                fill_shape = list(batch_shape)
                fill_shape[0] = paddle.shape(self.loc + self.scale)[0].item()
                fill_dtype = (self.loc + self.scale).dtype
                zero_tmp = paddle.full(fill_shape, 0.0, fill_dtype)
            else:
                zero_tmp = paddle.full(batch_shape, 0.0, self.dtype)
            return paddle.add(
                0.5 + zero_tmp,
                0.5 * math.log(2 * math.pi) + paddle.log(self.scale + zero_tmp),
                name=name,
            )

    def log_prob(self, value: Tensor) -> Tensor:
        """Log probability density/mass function.

        Args:
          value (Tensor): The input tensor.

        Returns:
          Tensor: log probability.The data type is same with :attr:`value` .

        """
        name = self.name + '_log_prob'
        value = self._check_values_dtype_in_probs(self.loc, value)

        var = self.scale * self.scale
        log_scale = paddle.log(self.scale)
        if self._complex_gaussian:
            return paddle.subtract(
                -1.0 * ((value - self.loc).conj() * (value - self.loc)) / (var),
                2.0 * log_scale + math.log(math.pi),
                name=name,
            )
        else:
            return paddle.subtract(
                -1.0 * ((value - self.loc) * (value - self.loc)) / (2.0 * var),
                log_scale + math.log(math.sqrt(2.0 * math.pi)),
                name=name,
            )

    def probs(self, value: Tensor) -> Tensor:
        """Probability density/mass function.

        Args:
            value (Tensor): The input tensor.

        Returns:
            Tensor, probability. The data type is same with :attr:`value` .

        """
        name = self.name + '_probs'
        value = self._check_values_dtype_in_probs(self.loc, value)

        var = self.scale * self.scale
        if self._complex_gaussian:
            return paddle.divide(
                paddle.exp(
                    -1.0
                    * ((value - self.loc).conj() * (value - self.loc))
                    / (var)
                ),
                (math.pi * var),
                name=name,
            )
        else:
            return paddle.divide(
                paddle.exp(
                    -1.0
                    * ((value - self.loc) * (value - self.loc))
                    / (2.0 * var)
                ),
                (math.sqrt(2 * math.pi) * self.scale),
                name=name,
            )

    def kl_divergence(self, other: Normal) -> Tensor:
        r"""The KL-divergence between two normal distributions.

        If non-complex, the KL-divergence is

        .. math::

            KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = 0.5 (ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio})

        If complex gaussian:

        .. math::

            KL\_divergence(\mu_0, \sigma_0; \mu_1, \sigma_1) = ratio^2 + (\frac{diff}{\sigma_1})^2 - 1 - 2 \ln {ratio}

        .. math::

            ratio = \frac{\sigma_0}{\sigma_1}

        .. math::

            diff = \mu_1 - \mu_0

        In the above equation:

        * :math:`loc = \mu_0`: is the mean of current Normal distribution.
        * :math:`scale = \sigma_0`: is the std of current Normal distribution.
        * :math:`loc = \mu_1`: is the mean of other Normal distribution.
        * :math:`scale = \sigma_1`: is the std of other Normal distribution.
        * :math:`ratio`: is the ratio of scales.
        * :math:`diff`: is the difference between means.

        Args:
            other (Normal): instance of Normal.

        Returns:
            Tensor, kl-divergence between two normal distributions.The data type is float32.

        """
        if not in_dynamic_mode():
            check_type(other, 'other', Normal, 'kl_divergence')

        if self._complex_gaussian != other._complex_gaussian:
            raise ValueError(
                "The kl divergence must be computed between two distributions in the same number field."
            )
        name = self.name + '_kl_divergence'
        var_ratio = self.scale / other.scale
        var_ratio = var_ratio * var_ratio
        t1 = (self.loc - other.loc) / other.scale
        if self._complex_gaussian:
            t1 = t1.conj() * t1
            return var_ratio + t1 - 1.0 - paddle.log(var_ratio)
        else:
            t1 = t1 * t1
            return paddle.add(
                0.5 * var_ratio,
                0.5 * (t1 - 1.0 - paddle.log(var_ratio)),
                name=name,
            )
