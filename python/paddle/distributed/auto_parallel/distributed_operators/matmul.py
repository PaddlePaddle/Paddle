# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from .common import ShardTag
from .common import DistributedOperator
from .common import DistributedOperatorImpl
from .common import OperatorDistributedSignature
from .common import register_distributed_operator
from .common import register_distributed_operator_impl


class DistributedMatmul(DistributedOperator):
    def __init__(self, name):
        super(DistributedMatmul, self).__init__()
        self._name = name


register_distributed_operator("matmul", DistributedMatmul("matmul"))


# ColumnParallel
class DistributedMatmulImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl0, self).__init__()
        self._name = name
        self._dist_signature = OperatorDistributedSignature()

        # Declare valid proc_mesh_ndim
        self._dist_signature.add_valid_proc_mesh_ndim(1)
        self._dist_signature.add_valid_proc_mesh_ndim(2)

        # Declare valid shard strategy for dims of inputs and outputs
        self._dist_signature.set_valid_input_dim_shard(
            name='X', dim=-1, tag=ShardTag.Replicate)

        self._dist_signature.set_valid_input_dim_shard(
            name='Y', dim=0, tag=ShardTag.Replicate)
        self._dist_signature.set_valid_input_dim_shard(
            name='Y', dim=1, tag=ShardTag.Split)

        self._dist_signature.set_valid_output_dim_shard(
            name='Out', dim=-1, tag=ShardTag.Split)

        # Declare dims of inputs and outputs using same the shard strategy
        self._dist_signature.add_valid_inputs_same_shard_dims(
            [('input', 'X', -1), ('input', 'Y', 0)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'X', -2), ('output', 'Out', -2)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'X', 0), ('output', 'Out', 0)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'Y', 1), ('output', 'Out', -1)])


# RowParallel
class DistributedMatmulImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl1, self).__init__()
        self._name = name
        self._dist_signature = OperatorDistributedSignature()

        # Declare valid proc_mesh_ndim
        self._dist_signature.add_valid_proc_mesh_ndim(1)
        self._dist_signature.add_valid_proc_mesh_ndim(2)

        # Declare valid shard strategy for dims of inputs and outputs
        self._dist_signature.set_valid_input_dim_shard(
            name='X', dim=-1, tag=ShardTag.Split)

        self._dist_signature.set_valid_input_dim_shard(
            name='Y', dim=0, tag=ShardTag.Split)
        self._dist_signature.set_valid_input_dim_shard(
            name='Y', dim=1, tag=ShardTag.Replicate)

        self._dist_signature.set_valid_output_dim_shard(
            name='Out', dim=-1, tag=ShardTag.Replicate)

        # Declare dims of inputs and outputs using same the shard strategy
        self._dist_signature.add_valid_inputs_same_shard_dims(
            [('input', 'X', -1), ('input', 'Y', 0)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'X', 0), ('output', 'Out', 0)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'X', -2), ('output', 'Out', -2)])
        self._dist_signature.add_valid_inputs_outputs_same_shard_dims(
            [('input', 'Y', 1), ('output', 'Out', -1)])


register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl0("column_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("row_parallel"))
