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


class DistributedMatmul(DistributedOperator):
    def __init__(self, name):
        super(DistributedMatmul, self).__init__()
        self._name = name


register_distributed_operator("matmul", DistributedMatmul("matmul"))


def _update_dims_mapping_for_matmul_new(op_dist_attr):
    changed = False
    op_desc = op_dist_attr.get_desc()
    x_name = op_desc.input('X')[0] 
    y_name = op_desc.input('Y')[0] 
    out_name = op_desc.output('Out')[0] 
    x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
    y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
    out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
    x_dims_mapping_len = len(x_dims_mapping)
    y_dims_mapping_len = len(y_dims_mapping)
    out_dims_mapping_len = len(out_dims_mapping)

    if x_dims_mapping_len == 1:
        x_dims_mapping.insert(0, -1)
    if y_dims_mapping_len == 1:
        x_dims_mapping.insert(1, -1)

    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, y_dims_mapping], [-1, 0])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, out_dims_mapping], [-2, -2])
    if dim_changed:
        changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [y_dims_mapping, out_dims_mapping], [1, -1])
    if dim_changed:
        changed = True
    
    if out_dims_mapping_len > 2:
        if x_dims_mapping_len < out_dims_mapping_len:
            for i in range(out_dims_mapping_len - x_dims_mapping_len):
                x_dims_mapping.insert(i, y_dims_mapping[i])
        if y_dims_mapping_len < out_dims_mapping_len:
            for i in range(out_dims_mapping_len - y_dims_mapping_len):
                y_dims_mapping.insert(i, y_dims_mapping[i])
        
        compatible_dims_mapping = compute_compatible_dims_mapping([x_dims_mapping,
                                                                   y_dims_mapping,
                                                                   out_dims_mapping])
        assert compatible_dims_mapping is not None, "There is no compatible dim mapping."

        for i in range(len(compatible_dims_mapping)):
            if x_dims_mapping[i] != compatible_dims_mapping[i]:
                x_dims_mapping[i] = compatible_dims_mapping[i] 
                changed = True
            if y_dims_mapping[i] != compatible_dims_mapping[i]:
                y_dims_mapping[i] = compatible_dims_mapping[i] 
                changed = True
            if out_dims_mapping[i] != compatible_dims_mapping[i]:
                out_dims_mapping[i] = compatible_dims_mapping[i] 
                changed = True
    
        if x_dims_mapping_len < out_dims_mapping_len:
            for i in range(out_dims_mapping_len - x_dims_mapping_len):
                x_dims_mapping.pop(i)
        if y_dims_mapping_len < out_dims_mapping_len:
            for i in range(out_dims_mapping_len - y_dims_mapping_len):
                y_dims_mapping.pop(i)

    if x_dims_mapping_len == 1:
        x_dims_mapping.pop(0)
    if y_dims_mapping_len == 1:
        x_dims_mapping.pop(1)
    
    return changed


def _update_dims_mapping_for_matmul(op_dist_attr):
    changed = False
    op_desc = op_dist_attr.get_desc()
    x_name = op_desc.input('X')[0] 
    y_name = op_desc.input('Y')[0] 
    out_name = op_desc.output('Out')[0] 
    x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
    y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
    out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, y_dims_mapping], [-1, 0])
    if dim_changed:
        changed = True

    if is_valid_list_index(x_dims_mapping, -2) and is_valid_list_index(out_dims_mapping, -2):
        dim_changed = compute_compatible_and_update_dim_mapping(
            [x_dims_mapping, out_dims_mapping], [-2, -2])
        if dim_changed:
            changed = True

    dim_changed = compute_compatible_and_update_dim_mapping(
        [y_dims_mapping, out_dims_mapping], [1, -1])
    if dim_changed:
        changed = True
    
    # Batch dimension
    dim_changed = compute_compatible_and_update_dim_mapping(
        [x_dims_mapping, out_dims_mapping], [0, 0])
    if dim_changed:
        changed = True

    return changed


# ColumnParallel
class DistributedMatmulImpl0(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl0, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        process_mesh = op_desc.get_process_mesh()
        if process_mesh.get_ndim() in [1, 2]:
            return True
    
    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        x_name = op_desc.input('X')[0] 
        y_name = op_desc.input('Y')[0] 
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_dim_shard(y_dims_mapping[0]) or is_dim_replicate(y_dims_mapping[1]):
            return False
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True 

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        out_name = op_desc.output('Out')[0] 
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_replicate(out_dims_mapping[-1]):
            return False
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True 
    
    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
        if dim_changed:
            changed = True
        return changed 
        

# RowParallel
class DistributedMatmulImpl1(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl1, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        process_mesh = op_desc.get_process_mesh()
        if process_mesh.get_ndim() in [1, 2]:
            return True
    
    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        x_name = op_desc.input('X')[0] 
        y_name = op_desc.input('Y')[0] 
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)
        if is_dim_replicate(x_dims_mapping[-1]):
            return False
        if is_dim_replicate(y_dims_mapping[0]) or is_dim_shard(y_dims_mapping[1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in x_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True 

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        out_name = op_desc.output('Out')[0] 
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)
        if is_dim_shard(out_dims_mapping[-1]):
            return False
        # Other dimensions must be replicate except the batch dimension
        for mapping in out_dims_mapping[1:-1]:
            if is_dim_shard(mapping):
                return False
        return True 
    
    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
        if dim_changed:
            changed = True
        return changed 



# ReplicateParallel 
class DistributedMatmulImpl2(DistributedOperatorImpl):
    def __init__(self, name):
        super(DistributedMatmulImpl2, self).__init__()
        self._name = name

    def is_process_mesh_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        process_mesh = op_desc.get_process_mesh()
        if process_mesh.get_ndim() in [1, 2]:
            return True
    
    def is_input_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        x_name = op_desc.input('X')[0] 
        y_name = op_desc.input('Y')[0] 
        x_dims_mapping = op_dist_attr.get_input_dims_mapping(x_name)
        y_dims_mapping = op_dist_attr.get_input_dims_mapping(y_name)

        if is_dim_shard(x_dims_mapping[-1]):
            return False
        if is_valid_list_index(x_dims_mapping, -2) and is_dim_shard(x_dims_mapping[-2]):
            return False

        if is_dim_shard(y_dims_mapping[-1]):
            return False
        if is_valid_list_index(y_dims_mapping, -2) and is_dim_shard(y_dims_mapping[-2]):
            return False

        return True 

    def is_output_compatible(self, op_dist_attr):
        op_desc = op_dist_attr.get_desc()
        out_name = op_desc.output('Out')[0] 
        out_dims_mapping = op_dist_attr.get_output_dims_mapping(out_name)

        if is_dim_shard(out_dims_mapping[-1]):
            return False
        if is_valid_list_index(out_dims_mapping, -2) and is_dim_shard(out_dims_mapping[-2]):
            return False

        return True 
    
    def update_dims_mapping(self, op_dist_attr):
        changed = False
        dim_changed = _update_dims_mapping_for_matmul(op_dist_attr)
        if dim_changed:
            changed = True
        return changed 


register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl0("column_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("row_parallel"))
register_distributed_operator_impl("matmul",
                                   DistributedMatmulImpl1("replicate_parallel"))
