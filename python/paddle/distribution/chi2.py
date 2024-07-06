# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from typing import TYPE_CHECKING

import paddle
from paddle.base.data_feeder import check_type, convert_dtype
from paddle.base.framework import Variable
from paddle.distribution.gamma import Gamma
from paddle.framework import in_dynamic_mode

if TYPE_CHECKING:
    from paddle import Tensor, dtype

__all__ = ["Chi2"]


class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter.
    This is exactly equivalent to Gamma(concentration=0.5*df, rate=0.5), :ref:`api_paddle_distribution_Gamma`.

    Args:
        df (float or Tensor): The degree of freedom of the distribution, which should be non-negative. If the input data type is Tensor, it indicates the batch creation of distributions with multiple different parameters, and the `batch_shape` (refer to the :ref:`api_paddle_distribution_Distribution` base class) is the parameter.

    Example:
        .. code-block:: python

            >>> import paddle
            >>> m = paddle.distribution.Chi2(paddle.to_tensor([1.0]))
            >>> sample = m.sample()
            >>> sample.shape
            [1]

    """

    df: Tensor
    rate: Tensor
    dtype: dtype

    def __init__(self, df: float | Tensor) -> None:
        if not in_dynamic_mode():
            check_type(
                df,
                'df',
                (float, Variable, paddle.pir.Value),
                'Chi2',
            )

        # Get/convert concentration to tensor.
        if self._validate_args(df):
            self.df = df
            self.dtype = convert_dtype(df.dtype)
        else:
            [self.df] = self._to_tensor(df)
            self.dtype = paddle.get_default_dtype()

        self.rate = paddle.full_like(self.df, 0.5)

        if in_dynamic_mode():
            if not paddle.all(self.df > 0):
                raise ValueError("The arg of `df` must be positive.")

        super().__init__(self.df * 0.5, self.rate)
