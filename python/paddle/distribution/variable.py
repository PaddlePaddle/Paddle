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

from paddle.distribution import constraint


class Variable(object):
    """This class is used to describe random variable of probability 
    distribution.

    Args:
        is_discrete (bool): Is the variable discrete or continuous.
        event_rank (int): The rank of event dimensions.
    """

    def __init__(self, is_discrete=False, event_rank=0, constraint=None):
        self._is_discrete = is_discrete
        self._event_rank = event_rank
        self._constraint = constraint

    @property
    def is_discrete(self):
        return self._is_discrete

    @property
    def event_rank(self):
        return self._event_rank

    def constraint(self, value):
        """Check whether the 'value' meet the constraint conditions of this 
        random variable."""
        return self._constraint(value)


class Real(Variable):
    def __init__(self, event_rank=0):
        super(Real, self).__init__(False, event_rank, constraint.real)


class Positive(Variable):
    def __init__(self, event_rank=0):
        super(Positive, self).__init__(False, event_rank, constraint.positive)


class Independent(Variable):
    """Reinterprets some of the batch axes of variable as event axes.

    Args:
        base (Variable): Base variable.
        reinterpreted_batch_rank (int): The rightmost batch rank to be 
            reinterpreted. 
    """

    def __init__(self, base, reinterpreted_batch_rank):
        self._base = base
        self._reinterpreted_batch_ndims
        super(Variable, self).__init__(
            base.is_discrete, base.event_rank + reinterpreted_batch_rank)

    def constraint(self, value):
        ret = self.base.constraint(value)
        if ret.dim() < self.reinterpreted_batch_ndims:
            raise ValueError(
                "Input dimensions must be equal or grater than  {}".format(
                    self.reinterpreted_batch_ndims))
        return ret.reshape(ret.shape[:ret.dim(
        ) - self.reinterpreted_batch_ndims] + (-1, )).all(-1)


# class _Stack(_Variable):
#     def __init__(self, constraints, axis=0):
#         self._constraints = constraints
#         self._axis = axis

#     @property
#     def is_discrete(self):
#         return any(c.is_discrete for c in self._constraints)

#     @property
#     def event_dim(self):
#         dim = max(c.event_dim for c in self._constraints)
#         if self.dim + dim < 0:
#             dim += 1
#         return dim

#     def validate(self, v):
#         if not (-v.dim() <= self.dim < v.dim()):
#             raise ValueError(
#                 f'Input dimensions {v.dim()} should be grater than stack '
#                 f'constraint axis {self._axis}.')

#         return paddle.stack(
#             [c.check(v) for c, v in zip(
#                 self._constraints, paddle.unstack(v, self._axis))],
#             self._axis)

real = Real()
positive = Positive()
