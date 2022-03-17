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
