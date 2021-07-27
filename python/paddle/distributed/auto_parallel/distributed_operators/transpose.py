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


class DistributedTranspose2(DistributedOperator):
    def __init__(self, name):
        super(DistributedTranspose2, self).__init__()
        self._name = name


register_distributed_operator("transpose2", DistributedTranspose2("transpose2"))


class DistributedTranspose2Impl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedTranspose2Impl0, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        process_mesh = op_dist_attr.get_process_mesh()
        if process_mesh.get_ndim() in [1, 2]:
            return True
        else:
            False

    def is_input_compatible(self, op_dist_attr):
        return True

    def is_output_compatible(self, op_dist_attr):
        return True

    def update_dims_mapping(self, op_dist_attr):
        changed = False
        op_desc = op_dist_attr.get_desc()
        x_name = op_desc.input('X')[0]
        out_name = op_desc.output('Out')[0]
        x_shape_name = op_desc.output('XShape')[0]
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        x_shape_dims_mapping = op_dist_attr.get_output_dims_mapping(
            x_shape_name)
        perm = op_desc.attr('axis')
        # print("transpose2 perm", perm)

        assert len(x_dims_mapping) == len(perm)

        new_dims_mapping = [-1 for i in range(len(x_dims_mapping))]
        for i in range(len(x_dims_mapping)):
            new_dims_mapping[i] = x_dims_mapping[perm[i]]

        for i in range(len(out_dims_mapping)):
            dim_changed = compute_compatible_and_update_dim_mapping(
                [new_dims_mapping, out_dims_mapping], [i, i])
            if dim_changed:
                changed = True

        for i in range(len(x_dims_mapping)):
            if x_dims_mapping[perm[i]] != new_dims_mapping[i]:
                x_dims_mapping[perm[i]] = new_dims_mapping[i]
                changed = True

        for i in range(len(x_dims_mapping)):
            x_shape_dims_mapping[i + 1] = x_dims_mapping[i]

        return changed


register_distributed_operator_impl(
    "transpose2", DistributedTranspose2Impl0("same_mapping_transpose"))
