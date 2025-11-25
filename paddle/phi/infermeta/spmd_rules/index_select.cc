/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/index_select.h"

#include <unordered_set>
#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

static inline std::vector<int64_t> FilterIndexMeshDims(
    const std::vector<int64_t>& index_mesh_dims,
    const std::vector<std::vector<int64_t>>& x_dims_mapping,
    int axis,
    int mesh_ndim) {
  std::unordered_set<int64_t> conflict_dims;
  conflict_dims.reserve(mesh_ndim);
  for (int i = 0; i < static_cast<int>(x_dims_mapping.size()); ++i) {
    if (i == axis) continue;
    for (int64_t d : x_dims_mapping[static_cast<size_t>(i)]) {
      conflict_dims.insert(d);
    }
  }
  std::vector<int64_t> kept_dims;
  kept_dims.reserve(index_mesh_dims.size());
  for (int64_t d : index_mesh_dims) {
    if (conflict_dims.find(d) == conflict_dims.end()) {
      kept_dims.emplace_back(d);
    } else {
      VLOG(4) << "Conflict detected on mesh dim " << d
              << ". Replicating the index tensor.";
    }
  }
  return kept_dims;
}

SpmdInfo IndexSelectInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& index,
                              int axis) {
  // Step0: Verify Input
  EXTRACT_SHAPE_AND_DIST_ATTR_CO_SHARD(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_CO_SHARD(index);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  std::vector<std::vector<int64_t>> x_dims_mapping = x_dims_mapping_src;
  std::vector<std::vector<int64_t>> index_dims_mapping = index_dims_mapping_src;
  x_dims_mapping[axis].clear();
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  const std::vector<int64_t> filtered_index_mesh_dims =
      FilterIndexMeshDims(index_dims_mapping[0],
                          x_dims_mapping,
                          axis,
                          x_dist_attr_src.process_mesh().ndim());

  std::vector<std::vector<int64_t>> out_dims_mapping = x_dims_mapping;
  out_dims_mapping[axis] = filtered_index_mesh_dims;
  index_dims_mapping[0] = filtered_index_mesh_dims;
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "IndexSelectInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_OUTPUT(out_dist_attr_dst);
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo IndexSelectGradInferSpmd(const DistMetaTensor& x,
                                  const DistMetaTensor& index,
                                  const DistMetaTensor& out_grad,
                                  int axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR_CO_SHARD(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_CO_SHARD(index);
  EXTRACT_SHAPE_AND_DIST_ATTR_CO_SHARD(out_grad);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      common::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));
  PADDLE_ENFORCE_EQ(x_ndim,
                    out_grad_ndim,
                    common::errors::InvalidArgument(
                        "IndexSelectGrad: The rank of x [%d] and out_grad [%d] "
                        "must be the same.",
                        x_ndim,
                        out_grad_ndim));
  // now use forward spmd rule to reduce complexity without actual cost eval.
  SpmdInfo fwd_spmd_info = IndexSelectInferSpmd(x, index, axis);
  const TensorDistAttr& x_dist_attr_dst =
      PADDLE_GET_CONST(TensorDistAttr, fwd_spmd_info.first[0]);
  const TensorDistAttr& index_dist_attr_dst =
      PADDLE_GET_CONST(TensorDistAttr, fwd_spmd_info.first[1]);
  const TensorDistAttr& out_grad_dist_attr_dst =
      PADDLE_GET_CONST(TensorDistAttr, fwd_spmd_info.second[0]);

  TensorDistAttr x_grad_dist_attr_dst = x_dist_attr_dst;
  x_grad_dist_attr_dst.clean_partial_status();
  std::vector<int64_t> partial_dims =
      index_dist_attr_dst.multi_dims_mapping()[0];
  if (!partial_dims.empty()) {
    x_grad_dist_attr_dst.set_partial_status(partial_dims);
    VLOG(4) << "x_grad is marked as partial on mesh dim: "
            << str_join(partial_dims);
  }

  VLOG(4) << "IndexSelectGradInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr_dst);
  return {{x_dist_attr_dst, index_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}

}  // namespace phi::distributed
