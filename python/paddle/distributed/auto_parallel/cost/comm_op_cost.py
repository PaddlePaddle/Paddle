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

from .base_cost import register_op_cost


class CommContext:
    _instance = None
    _has_instance = False

    def __init__(self, cluster):
        if CommContext._has_instance:
            return
        self._alpha, self._beta = self.init_comm_args(cluster)

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            _has_instance = True
        return cls._instance

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    def init_comm_args(self, cluster):
        alpha = 0
        beta = {}
        return aplha, beta


class CommOpCost(OpCost):
    COMM_OP_TYPE = [
        "send_v2", "recv_v2", "c_broadcast", "c_allgather", "c_allreduce_sum"
    ]

    def __init__(self, op, dist_context=None, cluster=None):
        self._check_op_type(op)
        super(CommOpCost, self).__init__(op, dist_context, cluster)
        self._comm_context = CommContext(self.cluster)

    @property
    def comm_context(self):
        return self._comm_context

    def _check_comm_op_type(op):
        if op.type not in COMM_OP_TYPE:
            raise TypeError("Please Check op type in {}, but got {}.".format(
                COMM_OP_TYPE, op.type))

    def calc_cost(self):
        # For comm op, its memory cost is 0, flops is 0 in default
        cost = Cost()
        return cost


@register_op_cost
class AllreduceSumCost(CommOpCost):
    OP_TYPE = "c_allreduce_sum"

    def __init__(self, op, dist_context, cluster):
        super(AllreduceSumCost, self).__init__(op, dist_context, cluster)
        self._cost = self.calc_cost()

    def calc_time(self):
        return 0

    def calc_cost(self):
        cost = Cost()
        cost.time = calc_time()
        return cost
