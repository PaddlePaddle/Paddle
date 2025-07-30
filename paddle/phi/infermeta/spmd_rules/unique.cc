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

SpmdInfo UniqueInferSpmd(const DistMetaTensor& x,
                         bool return_index,
                         bool return_inverse,
                         bool return_counts,
                         const std::vector<int>& axis,
                         DataType dtype) {
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

  TensorDistAttr indices_dist_attr_dst = TensorDistAttr();
  if (return_index) {
    indices_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
    indices_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});
  }

  TensorDistAttr inverse_dist_attr_dst = TensorDistAttr();
  if (return_inverse) {
    inverse_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
    inverse_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});
    // TODO(dev): https://github.com/PaddlePaddle/Paddle/issues/72822
    // if (axis.empty()) {
    //   inverse_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
    // }
  }

  TensorDistAttr counts_dist_attr_dst = TensorDistAttr();
  if (return_counts) {
    counts_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
    counts_dist_attr_dst.set_dims_mapping(std::vector<int64_t>{-1});
  }

  VLOG(4) << "UniqueInferSpmd: All input and output TensorDistAttr are set to "
             "fully replicated status.";
  return {{x_dist_attr_dst},
          {out_dist_attr_dst,
           indices_dist_attr_dst,
           inverse_dist_attr_dst,
           counts_dist_attr_dst}};
}

SpmdInfo UniqueInferSpmdStatic(const DistMetaTensor& x,
                               bool return_index,
                               bool return_inverse,
                               bool return_counts,
                               const std::vector<int>& axis,
                               DataType dtype,
                               bool is_sorted) {
  return UniqueInferSpmd(
      x, return_index, return_inverse, return_counts, axis, dtype);
}
}  // namespace distributed
}  // namespace phi
