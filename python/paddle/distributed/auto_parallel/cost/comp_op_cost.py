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

from .base_cost import Cost, register_op_cost, CompOpCost, OP_COST_FACTORY


@register_op_cost
class MatmulV2OpCost(CompOpCost):
    OP_TYPE = "matmul_v2"

    def __init__(self, op=None, op_desc=None, cluster=None):
        super(OP_COST_FACTORY["matmul_v2"], self).__init__(
            op=op, op_desc=op_desc, cluster=cluster)

    # For a concrete COMP OP, the calc_time and calc_flops function needs to be overrided
    def calc_flops(self):
        # NOTE: The actual formula will be filled in the future
        return 0

    def calc_time(self):
        # NOTE: The actual formula will be filled in the future
        return 0
