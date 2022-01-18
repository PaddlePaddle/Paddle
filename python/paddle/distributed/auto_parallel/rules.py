#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import copy


def compute_compatible_process_mesh(process_mesh_list):
    """Compute the compatible process mesh given a list of process meshes."""
    if not process_mesh_list:
        return None

    def _compute_compatible_process_mesh_two(pm1, pm2):
        if pm1 is None:
            return True, pm2
        if pm2 is None:
            return True, pm1
        if pm1 == pm2:
            return True, pm1
        if pm1.processes == pm2.processes:
            if len(pm1.topology) >= len(pm2.topology):
                return True, pm1
            else:
                return True, pm2
        process_set1 = set(pm1.processes)
        process_set2 = set(pm2.processes)
        if process_set1.issubset(process_set2):
            return True, pm2
        if process_set2.issubset(process_set1):
            return True, pm1
        return False, None

    compatible_result = None
    for process_mesh in process_mesh_list:
        compatible, compatible_result = _compute_compatible_process_mesh_two(
            compatible_result, process_mesh)
        if not compatible:
            return None
    return copy.deepcopy(compatible_result)


def compute_compatible_dim_mapping(dim_mapping_list):
    """Compute the compatible dim mapping given a list of dim mapping."""
    if not dim_mapping_list:
        return None

    def _compute_compatible_dim_mapping_two(dm1, dm2):
        if dm1 == -1:
            return True, dm2
        if dm2 == -1:
            return True, dm1
        if dm1 == dm2:
            return True, dm1
        return False, None

    compatible_result = -1
    for mapping in dim_mapping_list:
        compatible, compatible_result = _compute_compatible_dim_mapping_two(
            compatible_result, mapping)
        if not compatible:
            return None
    return compatible_result


def compute_compatible_dims_mapping(dims_mapping_list):
    """Compute the compatible dims mapping given a list of dims mapping.
       Each of dims mapping is also a list.
    """
    if not dims_mapping_list:
        return None
    length = len(dims_mapping_list[0])
    for dims_mapping in dims_mapping_list:
        if dims_mapping is None:
            return None
        if len(dims_mapping) != length:
            return None
    compatible_result = []
    for dim_mappings in zip(*dims_mapping_list):
        compatible_dim_mapping = compute_compatible_dim_mapping(
            list(dim_mappings))
        if compatible_dim_mapping is None:
            return None
        compatible_result.append(compatible_dim_mapping)
    return compatible_result
