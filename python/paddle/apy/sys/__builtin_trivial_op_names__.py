# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


def GetGroupedTrivialOpNames():
    return [
        "pd_op.sin",
        "pd_op.add",
        "pd_op.relu",
        "pd_op.data",
        "pd_op.full",
        "pd_op.cast",
        "pd_op.exp",
        "pd_op.relu",
        "pd_op.tanh",
        "pd_op.floor",
        "pd_op.erf",
        "pd_op.elementwise_pow",
        "cinn_op.scale",
        "pd_op.subtract",
        "pd_op.add",
        "pd_op.multiply",
        "pd_op.divide",
        "pd_op.maximum",
        "cinn_op.yield_store",
        "cinn_op.broadcast",
        "pd_op.expand",
        "cinn_op.generate_shape",
    ]
