#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
# limitations under the License


class Cost:
    def __init__(self, time=0., memory=0, flops=0):
        self.time = time
        self.memory = memory
        self.flops = flops

    def _check_time(self, val):
        assert isinstance(
            val, float
        ) and val >= 0, "Time must be float and greater than or equal to 0."

    def _check_memory(self, val):
        assert isinstance(
            val, int) and val >= 0, "Memory must be int and greater than 0."

    def _check_flops(self, val):
        assert isinstance(
            val, int) and val >= 0, "FLOPs must be int and greater than 0."

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, val):
        self._check_time(val)
        self._time = val

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, val):
        self._check_memory(val)
        self._memory = val

    @property
    def flops(self):
        return self._flops

    @flops.setter
    def flops(self, val):
        self._check_flops(val)
        self._flops = val

    def __add__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time + rhs.time
        memory = self.memory + rhs.memory
        flops = self.flops + rhs.flops
        return Cost(time, memory, flops)

    def __sub__(self, rhs):
        assert isinstance(rhs, Cost)
        time = self.time - rhs.time
        memory = self.memory - rhs.memory
        flops = self.flops - rhs.flops
        return Cost(time, memory, flops)


class OpCost:
    def __init__(self, op, dist_context=None, cluster=None):
        self._op = op
        self._dist_context = dist_context
        self._cluster = cluster
        self._cost = None

    @property
    def op(self):
        return self._op

    @property
    def cost(self):
        return self._cost

    def calc_cost(self):
        raise NotImplementedError


OP_COST_FACTORY = {}


def register_op_cost(cls):
    op_type = cls.OP_TYPE

    def register(op_type):
        OP_COST_FACTORY[op_type] = cls

    return register(op_type)
