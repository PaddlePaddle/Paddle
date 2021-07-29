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

from .common import DistributedOperator
from .common import DistributedOperatorImpl
from .common import register_distributed_operator
from .common import register_distributed_operator_impl
from ..utils import is_dim_shard
from ..utils import is_dim_replicate
from ..utils import is_valid_list_index
from ..utils import compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping
from ..utils import compute_compatible_and_update_dim_mapping


class DistributedEmbedding(DistributedOperator):
    def __init__(self, name):
        super(DistributedEmbedding, self).__init__()
        self._name = name


register_distributed_operator("lookup_table_v2",
                              DistributedEmbedding("embedding"))


# RowParallel
class DistributedEmbeddingImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedEmbeddingImpl0, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        # print("process_mesh_compatible")
        process_mesh = op_dist_attr.get_process_mesh()
        if process_mesh.get_ndim() in [1, 2]:
            return True
        else:
            False

    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        if is_dim_replicate(w_dims_mapping[-2]) or is_dim_shard(w_dims_mapping[
                -1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in ids_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        out_name = op_desc.output('Out')[0]
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:]:
            if is_dim_shard(mapping):
                return False
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        op_desc = op_dist_attr.get_desc()
        ids_name = op_desc.input('Ids')[0]
        w_name = op_desc.input('W')[0]
        out_name = op_desc.output('Out')[0]
        ids_dims_mapping = op_dist_attr.get_input_dims_mapping(ids_name)
        w_dims_mapping = op_dist_attr.get_input_dims_mapping(w_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        for i in range(len(ids_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [ids_dims_mapping, out_dims_mapping], [i, i])
            if dim_changed:
                changed = True

        dim_changed = compute_compatible_and_update_dim_mapping(
            [w_dims_mapping, out_dims_mapping], [-1, -1])
        if dim_changed:
            changed = True

        return changed


register_distributed_operator_impl("lookup_table_v2",
                                   DistributedEmbeddingImpl0("row_parallel"))
