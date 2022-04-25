# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.distribution import distribution


class Independent(distribution.Distribution):
    r"""
    Reinterprets some of the batch dimensions of a distribution as event dimensions.

    This is mainly useful for changing the shape of the result of
    :meth:`log_prob`. 

    Args:
        base (Distribution): The base distribution.
        reinterpreted_batch_rank (int): The number of batch dimensions to 
            reinterpret as event dimensions.

    Examples:

        .. code-block:: python
        
            import paddle
            from paddle.distribution import independent

            beta = paddle.distribution.Beta(paddle.to_tensor([0.5, 0.5]), paddle.to_tensor([0.5, 0.5]))
            print(beta.batch_shape, beta.event_shape)
            # (2,) ()
            print(beta.log_prob(paddle.to_tensor(0.2)))
            # Tensor(shape=[2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [-0.22843921, -0.22843921])
            reinterpreted_beta = independent.Independent(beta, 1)
            print(reinterpreted_beta.batch_shape, reinterpreted_beta.event_shape)
            # () (2,)
            print(reinterpreted_beta.log_prob(paddle.to_tensor([0.2,  0.2])))
            # Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
            #        [-0.45687842])
    """

    def __init__(self, base, reinterpreted_batch_rank):
        if not isinstance(base, distribution.Distribution):
            raise TypeError(
                f"Expected type of 'base' is Distribution, but got {type(base)}")
        if not (0 < reinterpreted_batch_rank <= len(base.batch_shape)):
            raise ValueError(
                f"Expected 0 < reinterpreted_batch_rank <= {len(base.batch_shape)}, but got {reinterpreted_batch_rank}"
            )
        self._base = base
        self._reinterpreted_batch_rank = reinterpreted_batch_rank

        shape = base.batch_shape + base.event_shape
        super(Independent, self).__init__(
            batch_shape=shape[:len(base.batch_shape) -
                              reinterpreted_batch_rank],
            event_shape=shape[len(base.batch_shape) -
                              reinterpreted_batch_rank:])

    @property
    def mean(self):
        return self._base.mean

    @property
    def variance(self):
        return self._base.variance

    def sample(self, shape=()):
        return self._base.sample(shape)

    def log_prob(self, value):
        return self._sum_rightmost(
            self._base.log_prob(value), self._reinterpreted_batch_rank)

    def prob(self, value):
        return self.log_prob(value).exp()

    def entropy(self):
        return self._sum_rightmost(self._base.entropy(),
                                   self._reinterpreted_batch_rank)

    def _sum_rightmost(self, value, n):
        return value.sum(list(range(-n, 0))) if n > 0 else value
