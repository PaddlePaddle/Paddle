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

from .comm_op_cost import COMM_OP_TYPE
from .base_cost import Cost, register_op_cost

SPEC_OP_TYPE = ["while"] + COMM_OP_TYPE


def _parse_op_info(op):
    op_info = {}
    return op_info


def calc_time_from_benchmark(op_info):
    return 0


class CompOpCost(OpCost):
    def __init__(self, op, dist_context=None, cluster=None):
        self._check_op_type(op)
        super(CompOpCost, self).__init__(op, dist_context, cluster)

    def _check_comp_op_type(op):
        if op.type in SPEC_OP_TYPE:
            raise TypeError("Please Check op type not in {}, but got {}.".
                            format(SPEC_OP_TYPE, op.type))

    def calc_flops(self):
        raise NotImplementedError

    def calc_time(self):
        op_info = _parse_op_info(self.op)
        time = calc_time_from_benchmark(op_info)
        return time

    def calc_memory(self):
        return 0


@register_op_cost
class MatmulV2OpCost(CompOpCost):
    OP_TYPE = "matmul_v2"

    def __init__(self, op, dist_context=None, cluster=None):
        assert op.type == "matmul_v2"
        super(MatmulV2OpCost, self).__init__(op, dist_context, cluster)
        self._cost = self.calc_cost()

    def calc_flops(self):
        return 0

    def calc_cost(self):
        time = self.calc_time()
        memory = self.calc_memory()
        flops = self.calc_flops()
        cost = Cost(time, memory, flops)
        return cost
