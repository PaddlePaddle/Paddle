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

from .base_reshard_func import ReshardFunction


class GlobaleToSubMeshFunction(ReshardFunction):
    def is_suitable(self, src_dist_attr, dst_dist_attr):
        if 0 in src_dist_attr.dims_mapping or 0 in src_dist_attr.partial_status:
            return False
        in_mesh = src_dist_attr.process_mesh
        out_mesh = dst_dist_attr.process_mesh
        if in_mesh.ndim != out_mesh.ndim + 1:
            return False
        sub_meshes = paddle.base.libpaddle.pir.get_sub_meshes(in_mesh)
        return out_mesh in sub_meshes

    def reshard(self, src_dist_attr, dst_dist_attr, src_value, dst_type):
        if src_value.has_one_use():
            src_value.update_dist_attr(dst_dist_attr)
            prev_op = src_value.get_defining_op()
            op_dist_attr = prev_op.dist_attr
            op_mesh = op_dist_attr.process_mesh
            operands = op_dist_attr.operands()
            results = op_dist_attr.results()
            results[src_value.index()] = dst_dist_attr
            prev_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    op_mesh, operands, results
                )
            )
            return src_value
        else:
            dst_value = paddle._C_ops.share_data_(src_value)
            share_data_op = dst_value.get_defining_op()
            # set dist type and dist attr
            dst_value.set_type(dst_type)
            share_data_op.dist_attr = (
                paddle.base.libpaddle.pir.create_op_dist_attribute(
                    src_dist_attr.process_mesh, [src_dist_attr], [dst_dist_attr]
                )
            )
            return dst_value
