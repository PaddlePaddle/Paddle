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


class _Constraint(object):
    def __call__(self, value):
        raise NotImplementedError


class _Real(_Constraint):
    def __call__(self, v):
        return v == v


class _Interval(_Constraint):
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        super(_Interval, self).__init__()

    def __call__(self, v):
        return self._lower <= v <= self.upper


class _Positive(_Constraint):
    def __call__(self, v):
        return v >= 0.


class _Simplex(_Constraint):
    def __call__(self, v):
        return paddle.all(v >= 0, dim=-1) and ((v.sum(-1) - 1).abs() < 1e-6)


class _LowerCholesky(_Constraint):
    def __call__(self, v):
        pass


class _CorrelationCholesky(_Constraint):
    event_dim = 2

    def validate(self, v):
        pass


real = _Real()
positive = _Positive()
