# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

# all registered reshard functions
_g_reshard_func_list = []


class ReshardFunction:
    def is_suitable(self, dist_tensor, dist_attr):
        raise NotImplementedError

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        raise NotImplementedError


def choose_reshard_func(src_dist_attr, dst_dist_attr):
    global _g_reshard_func_list
    for reshard_func in _g_reshard_func_list:
        if reshard_func.is_suitable(src_dist_attr, dst_dist_attr):
            return reshard_func
    return None


def register_reshard_func(reshard_func):
    global _g_reshard_func_list
    _g_reshard_func_list.append(reshard_func)


def clean_reshard_funcs():
    global _g_reshard_func_list
    _g_reshard_func_list.clear()


def is_shard(dist_attr):
    for v in dist_attr.dims_mapping:
        if v != -1:
            return True
    return False


def is_partial(dist_attr):
    if len(dist_attr.partial_status) > 0:
        return True
    return False


def is_replicated(dist_attr):
    dims_mapping_set = set(dist_attr.dims_mapping)
    if len(dist_attr.partial_status) == 0 and (
        len(dims_mapping_set) == 0
        or (len(dims_mapping_set) == 1 and -1 in dims_mapping_set)
    ):
        return True
    return False


def copy_dist_attr_with_new_member(
    src_dist_attr,
    new_process_mesh=None,
    new_dims_mapping=None,
    new_partial_status=None,
):
    if new_process_mesh is None:
        new_process_mesh = src_dist_attr.process_mesh
    if new_dims_mapping is None:
        new_dims_mapping = src_dist_attr.dims_mapping
    if new_partial_status is None:
        new_partial_status = src_dist_attr.partial_status

    return paddle.base.libpaddle.pir.create_tensor_dist_attribute(
        new_process_mesh,
        new_dims_mapping,
        new_partial_status,
    )


def copy_op_attr_with_new_member(
    src_dist_attr,
    new_process_mesh=None,
    new_operands=None,
    new_results=None,
    new_chunk_id=None,
):
    if new_process_mesh is None:
        new_process_mesh = src_dist_attr.process_mesh
    if new_operands is None:
        new_operands = src_dist_attr.operands()
    if new_results is None:
        new_results = src_dist_attr.results()
    if new_chunk_id is None:
        new_chunk_id = src_dist_attr.chunk_id

    return paddle.base.libpaddle.pir.create_op_dist_attribute(
        new_process_mesh,
        new_operands,
        new_results,
        new_chunk_id,
    )


def copy_process_mesh_with_new_member(
    src_process_mesh,
    new_shape=None,
    new_process_ids=None,
    new_dim_names=None,
):
    if new_shape is None:
        new_shape = src_process_mesh.shape
    if new_process_ids is None:
        new_process_ids = src_process_mesh.process_ids
    if new_dim_names is None:
        new_dim_names = src_process_mesh.dim_names

    return paddle.base.libpaddle.pir.create_process_mesh(
        new_shape,
        new_process_ids,
        new_dim_names,
    )
