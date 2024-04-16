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
import paddle
from paddle.distribution.gamma import Gamma

__all__ = ["Chi2"]


class Chi2(Gamma):
    r"""
    Creates a Chi-squared distribution parameterized by shape parameter :attr:df.
    This is exactly equivalent to Gamma(concentration=0.5*df, rate=0.5)

    Example::
        .. code-block:: python
            >>> m = Chi2(paddle.to_tensor([1.0]))
            >>> m.sample()
            tensor([ 0.1046])

    Args:
        df (float or Tensor): shape parameter of the distribution
    """

    def __init__(self, df):
        if not isinstance(df, paddle.Tensor):
            df = paddle.to_tensor(df)
        super().__init__(0.5 * df, paddle.to_tensor(0.5))

    @property
    def df(self):
        return self.concentration * 2
