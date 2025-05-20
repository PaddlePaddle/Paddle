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

#include "paddle/phi/infermeta/spmd_rules/unique.h"
#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo UniqueInferSpmdBase(const DistMetaTensor& x,
                             bool return_index,
                             bool return_inverse,
                             bool return_counts,
                             const std::vector<int>& axis) {
  // Verify input args
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  std::vector<int64_t> x_dims_mapping_dst(x_ndim, -1);
  std::vector<int64_t> out_dims_mapping_dst(x_dims_mapping_dst);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  if (axis.empty()) {
    out_dims_mapping_dst = {-1};
  }
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(out_dims_mapping_dst);
  std::vector<TensorDistAttr> outputs_spmd_info = {out_dist_attr_dst};

  if (return_index) {
    TensorDistAttr indices_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    indices_dist_attr_dst.set_dims_mapping({-1});
    outputs_spmd_info.push_back(indices_dist_attr_dst);
  }

  if (return_inverse) {
    TensorDistAttr inverse_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    inverse_dist_attr_dst.set_dims_mapping({-1});
    outputs_spmd_info.push_back(inverse_dist_attr_dst);
  }

  if (return_counts) {
    TensorDistAttr counts_dist_attr_dst =
        CopyTensorDistAttrForOutput(x_dist_attr_src);
    counts_dist_attr_dst.set_dims_mapping({-1});
    outputs_spmd_info.push_back(counts_dist_attr_dst);
  }

  return {{x_dist_attr_dst}, ToArgDistAttr(outputs_spmd_info)};
}

SpmdInfo UniqueInferSpmd(const DistMetaTensor& x,
                         bool return_index,
                         bool return_inverse,
                         bool return_counts,
                         const std::vector<int>& axis,
                         DataType dtype) {
  return UniqueInferSpmdBase(
      x, return_index, return_inverse, return_counts, axis);
}

SpmdInfo UniqueInferSpmdStatic(const DistMetaTensor& x,
                               bool return_index,
                               bool return_inverse,
                               bool return_counts,
                               const std::vector<int>& axis,
                               DataType dtype,
                               bool is_sorted) {
  return UniqueInferSpmdBase(
      x, return_index, return_inverse, return_counts, axis);
}
}  // namespace distributed
}  // namespace phi
