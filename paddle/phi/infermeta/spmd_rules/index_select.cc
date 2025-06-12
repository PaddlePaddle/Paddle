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

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo IndexSelectInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& index,
                              int axis) {
  // Step0: Verify Input
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      phi::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  std::vector<int64_t> x_dims_mapping = x_dims_mapping_src;
  std::vector<int64_t> index_dims_mapping = index_dims_mapping_src;
  x_dims_mapping[axis] = -1;
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  std::vector<int64_t> out_dims_mapping(x_ndim, -1);
  int64_t index_mesh_dim = index_dims_mapping[0];
  for (int i = 0; i < x_ndim; ++i) {
    if (i != axis) {
      out_dims_mapping[i] = x_dims_mapping[i];
      // input shared usually more useful than index shared
      if (index_mesh_dim != -1 && out_dims_mapping[i] == index_mesh_dim) {
        VLOG(4) << "Conflict detected on mesh dim " << index_mesh_dim
                << ". Replicating the index tensor.";
        index_mesh_dim = -1;
        index_dims_mapping[0] = -1;
      }
    }
  }
  out_dims_mapping[axis] = index_mesh_dim;
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
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  axis = axis < 0 ? x_ndim + axis : axis;
  PADDLE_ENFORCE_EQ(
      0 <= axis && axis < x_ndim,
      true,
      phi::errors::InvalidArgument(
          "The axis of index_select should be in range [0, %d), but got %d.",
          x_ndim,
          axis));
  PADDLE_ENFORCE_EQ(x_ndim,
                    out_grad_ndim,
                    common::errors::InvalidArgument(
                        "IndexSelectGrad: The rank of x [%d] and outgrad [%d] "
                        "must be the same.",
                        x_ndim,
                        out_grad_ndim));
  // now use forward spmd rule to reduce complexity without actual cost eval.
  SpmdInfo fwd_spmd_info = IndexSelectInferSpmd(x, index, axis);
  TensorDistAttr x_dist_attr_dst = paddle::get<0>(fwd_spmd_info.first[0]);
  TensorDistAttr index_dist_attr_dst = paddle::get<0>(fwd_spmd_info.first[1]);
  TensorDistAttr out_grad_dist_attr_dst =
      paddle::get<0>(fwd_spmd_info.second[0]);

  TensorDistAttr x_grad_dist_attr_dst = x_dist_attr_dst;
  x_grad_dist_attr_dst.clean_partial_status();
  if (index_dist_attr_dst.dims_mapping()[0] != -1) {
    std::vector<int64_t> partial_dims(1, index_dist_attr_dst.dims_mapping()[0]);
    x_grad_dist_attr_dst.set_partial_status(partial_dims);
    VLOG(4) << "x_grad is marked as partial on mesh dim: " << partial_dims[0];
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
