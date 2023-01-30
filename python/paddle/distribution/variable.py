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

<<<<<<< HEAD
import paddle
from paddle.distribution import constraint


class Variable:
=======
from paddle.distribution import constraint


class Variable(object):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """Random variable of probability distribution.

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
<<<<<<< HEAD
        """Check whether the 'value' meet the constraint conditions of this
=======
        """Check whether the 'value' meet the constraint conditions of this 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        random variable."""
        return self._constraint(value)


class Real(Variable):
<<<<<<< HEAD
    def __init__(self, event_rank=0):
        super().__init__(False, event_rank, constraint.real)


class Positive(Variable):
    def __init__(self, event_rank=0):
        super().__init__(False, event_rank, constraint.positive)
=======

    def __init__(self, event_rank=0):
        super(Real, self).__init__(False, event_rank, constraint.real)


class Positive(Variable):

    def __init__(self, event_rank=0):
        super(Positive, self).__init__(False, event_rank, constraint.positive)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


class Independent(Variable):
    """Reinterprets some of the batch axes of variable as event axes.

    Args:
        base (Variable): Base variable.
<<<<<<< HEAD
        reinterpreted_batch_rank (int): The rightmost batch rank to be
            reinterpreted.
=======
        reinterpreted_batch_rank (int): The rightmost batch rank to be 
            reinterpreted. 
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    """

    def __init__(self, base, reinterpreted_batch_rank):
        self._base = base
        self._reinterpreted_batch_rank = reinterpreted_batch_rank
<<<<<<< HEAD
        super().__init__(
            base.is_discrete, base.event_rank + reinterpreted_batch_rank
        )
=======
        super(Independent,
              self).__init__(base.is_discrete,
                             base.event_rank + reinterpreted_batch_rank)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

    def constraint(self, value):
        ret = self._base.constraint(value)
        if ret.dim() < self._reinterpreted_batch_rank:
            raise ValueError(
                "Input dimensions must be equal or grater than  {}".format(
<<<<<<< HEAD
                    self._reinterpreted_batch_rank
                )
            )
        return ret.reshape(
            ret.shape[: ret.dim() - self.reinterpreted_batch_rank] + (-1,)
        ).all(-1)


class Stack(Variable):
=======
                    self._reinterpreted_batch_rank))
        return ret.reshape(ret.shape[:ret.dim() -
                                     self.reinterpreted_batch_rank] +
                           (-1, )).all(-1)


class Stack(Variable):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def __init__(self, vars, axis=0):
        self._vars = vars
        self._axis = axis

    @property
    def is_discrete(self):
        return any(var.is_discrete for var in self._vars)

    @property
    def event_rank(self):
        rank = max(var.event_rank for var in self._vars)
        if self._axis + rank < 0:
            rank += 1
        return rank

    def constraint(self, value):
        if not (-value.dim() <= self._axis < value.dim()):
            raise ValueError(
                f'Input dimensions {value.dim()} should be grater than stack '
<<<<<<< HEAD
                f'constraint axis {self._axis}.'
            )

        return paddle.stack(
            [
                var.check(value)
                for var, value in zip(
                    self._vars, paddle.unstack(value, self._axis)
                )
            ],
            self._axis,
        )
=======
                f'constraint axis {self._axis}.')

        return paddle.stack([
            var.check(value)
            for var, value in zip(self._vars, paddle.unstack(value, self._axis))
        ], self._axis)
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


real = Real()
positive = Positive()
