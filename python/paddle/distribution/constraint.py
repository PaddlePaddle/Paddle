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


class Constraint(object):
    def __call__(self, value):
        raise NotImplementedError


class Real(Constraint):
    def __call__(self, v):
        return v == v


class Range(Constraint):
    def __init__(self, lower, upper):
        self._lower = lower
        self._upper = upper
        super(Range, self).__init__()

    def __call__(self, v):
        return self._lower <= v <= self.upper


class Positive(Constraint):
    def __call__(self, v):
        return v >= 0.


class Simplex(Constraint):
    def __call__(self, v):
        return paddle.all(v >= 0, dim=-1) and ((v.sum(-1) - 1).abs() < 1e-6)


class LowerCholesky(Constraint):
    def __call__(self, v):
        pass


class CorrelationCholesky(Constraint):
    def __call__(self, v):
        pass


real = Real()
positive = Positive()
simplex = Simplex()
