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

# TODO: define the distribution functions
# __all__ = ['Categorical',
#            'MultivariateNormalDiag',
#            'Normal',
#            'Uniform']

import warnings

import numpy as np

import paddle
from paddle import _C_ops
from paddle.base.data_feeder import check_variable_and_dtype, convert_dtype
from paddle.base.framework import Variable
from paddle.framework import (
    in_dynamic_or_pir_mode,
    in_pir_mode,
)


class Distribution:
    """
    The abstract base class for probability distributions. Functions are
    implemented in specific distributions.

    Args:
        batch_shape(Sequence[int], optional):  independent, not identically
            distributed draws, aka a "collection" or "bunch" of distributions.
        event_shape(Sequence[int], optional): the shape of a single
            draw from the distribution; it may be dependent across dimensions.
            For scalar distributions, the event shape is []. For n-dimension
            multivariate distribution, the event shape is [n].
    """

    def __init__(self, batch_shape=(), event_shape=()):
        self._batch_shape = (
            batch_shape
            if isinstance(batch_shape, tuple)
            else tuple(batch_shape)
        )
        self._event_shape = (
            event_shape
            if isinstance(event_shape, tuple)
            else tuple(event_shape)
        )

        super().__init__()

    @property
    def batch_shape(self):
        """Returns batch shape of distribution

        Returns:
            Sequence[int]: batch shape
        """
        return self._batch_shape

    @property
    def event_shape(self):
        """Returns event shape of distribution

        Returns:
            Sequence[int]: event shape
        """
        return self._event_shape

    @property
    def mean(self):
        """Mean of distribution"""
        raise NotImplementedError

    @property
    def variance(self):
        """Variance of distribution"""
        raise NotImplementedError

    def sample(self, shape=()):
        """Sampling from the distribution."""
        raise NotImplementedError

    def rsample(self, shape=()):
        """reparameterized sample"""
        raise NotImplementedError

    def entropy(self):
        """The entropy of the distribution."""
        raise NotImplementedError

    def kl_divergence(self, other):
        """The KL-divergence between self distributions and other."""
        raise NotImplementedError

    def prob(self, value):
        """Probability density/mass function evaluated at value.

        Args:
            value (Tensor): value which will be evaluated
        """
        return self.log_prob(value).exp()

    def log_prob(self, value):
        """Log probability density/mass function."""
        raise NotImplementedError

    def probs(self, value):
        """Probability density/mass function.

        Note:

            This method will be deprecated in the future, please use `prob`
            instead.
        """
        raise NotImplementedError

    def _extend_shape(self, sample_shape):
        """compute shape of the sample

        Args:
            sample_shape (Tensor): sample shape

        Returns:
            Tensor: generated sample data shape
        """
        return (
            tuple(sample_shape)
            + tuple(self._batch_shape)
            + tuple(self._event_shape)
        )

    def _validate_args(self, *args):
        """
        Argument validation for distribution args
        Args:
            value (float, list, numpy.ndarray, Tensor)
        Raises
            ValueError: if one argument is Tensor, all arguments should be Tensor
        """
        is_variable = False
        is_number = False
        for arg in args:
            if isinstance(arg, (Variable, paddle.pir.Value)):
                is_variable = True
            else:
                is_number = True

        if is_variable and is_number:
            raise ValueError(
                'if one argument is Tensor, all arguments should be Tensor'
            )

        return is_variable

    def _to_tensor(self, *args):
        """
        Argument convert args to Tensor

        Args:
            value (float, list, numpy.ndarray, Tensor)
        Returns:
            Tensor of args.
        """
        numpy_args = []
        variable_args = []
        tmp = 0.0

        for arg in args:
            if not isinstance(
                arg,
                (float, list, tuple, np.ndarray, Variable, paddle.pir.Value),
            ):
                raise TypeError(
                    f"Type of input args must be float, list, tuple, numpy.ndarray or Tensor, but received type {type(arg)}"
                )
            if isinstance(arg, paddle.pir.Value):
                # pir.Value does not need to be converted to numpy.ndarray, so we skip here
                numpy_args.append(arg)
                continue

            arg_np = np.array(arg)
            arg_dtype = arg_np.dtype
            if str(arg_dtype) != 'float32':
                if str(arg_dtype) != 'float64':
                    # "assign" op doesn't support float64. if dtype is float64, float32 variable will be generated
                    #  and converted to float64 later using "cast".
                    warnings.warn(
                        "data type of argument only support float32 and float64, your argument will be convert to float32."
                    )
                arg_np = arg_np.astype('float32')
            # tmp is used to support broadcast, it summarizes shapes of all the args and get the mixed shape.
            tmp = tmp + arg_np
            numpy_args.append(arg_np)

        dtype = tmp.dtype
        for arg in numpy_args:
            if isinstance(arg, paddle.pir.Value):
                # pir.Value does not need to be converted to numpy.ndarray, so we skip here
                variable_args.append(arg)
                continue

            arg_broadcasted, _ = np.broadcast_arrays(arg, tmp)
            if in_pir_mode():
                arg_variable = paddle.zeros(arg_broadcasted.shape)
            else:
                arg_variable = paddle.tensor.create_tensor(dtype=dtype)
            paddle.assign(arg_broadcasted, arg_variable)
            variable_args.append(arg_variable)

        return tuple(variable_args)

    def _check_values_dtype_in_probs(self, param, value):
        """
        Log_prob and probs methods have input ``value``, if value's dtype is different from param,
        convert value's dtype to be consistent with param's dtype.

        Args:
            param (Tensor): low and high in Uniform class, loc and scale in Normal class.
            value (Tensor): The input tensor.

        Returns:
            value (Tensor): Change value's dtype if value's dtype is different from param.
        """
        if in_dynamic_or_pir_mode():
            if value.dtype != param.dtype and convert_dtype(value.dtype) in [
                'float32',
                'float64',
            ]:
                warnings.warn(
                    "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
                )
                return _C_ops.cast(value, param.dtype)
            return value

        check_variable_and_dtype(
            value, 'value', ['float32', 'float64'], 'log_prob'
        )
        if value.dtype != param.dtype:
            warnings.warn(
                "dtype of input 'value' needs to be the same as parameters of distribution class. dtype of 'value' will be converted."
            )
            return paddle.cast(value, dtype=param.dtype)
        return value

    def _probs_to_logits(self, probs, is_binary=False):
        r"""
        Converts probabilities into logits. For the binary, probs denotes the
        probability of occurrence of the event indexed by `1`. For the
        multi-dimensional, values of last axis denote the probabilities of
        occurrence of each of the events.
        """
        return (
            (paddle.log(probs) - paddle.log1p(-probs))
            if is_binary
            else paddle.log(probs)
        )

    def _logits_to_probs(self, logits, is_binary=False):
        r"""
        Converts logits into probabilities. For the binary, each value denotes
        log odds, whereas for the multi-dimensional case, the values along the
        last dimension denote the log probabilities of the events.
        """
        return (
            paddle.nn.functional.sigmoid(logits)
            if is_binary
            else paddle.nn.functional.softmax(logits, axis=-1)
        )
