/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/expand_as.h"

#include "glog/logging.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

std::tuple<TensorDistAttr, TensorDistAttr> AlignExpandAsDistAttrs(
    const DistMetaTensor& x, const DistMetaTensor& y) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(y);
  auto x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  auto y_dist_attr_dst = CopyTensorDistAttrForOutput(y_dist_attr_src);
  auto x_dims_mapping_dst = x_dims_mapping_src;
  const auto& y_dims_mapping_dst = y_dims_mapping_src;
  int dims_diff = y_ndim - x_ndim;
  for (int i = 0; i < y_ndim; ++i) {
    if (i >= dims_diff) {
      if (x_shape[i - dims_diff] == y_shape[i]) {
        x_dims_mapping_dst[i - dims_diff] = y_dims_mapping_src[i];
      } else {
        x_dims_mapping_dst[i - dims_diff] = -1;
      }
    }
  }
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  y_dist_attr_dst.set_dims_mapping(y_dims_mapping_dst);
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(y);
  return {x_dist_attr_dst, y_dist_attr_dst};
}

SpmdInfo ExpandAsInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& y,
                           const std::vector<int64_t>& target_shape) {
  auto [x_dist_attr, y_dist_attr] = AlignExpandAsDistAttrs(x, y);
  return {{x_dist_attr, y_dist_attr}, {y_dist_attr}};
}

SpmdInfo ExpandAsInferSpmdReverse(const DistMetaTensor& x,
                                  const DistMetaTensor& y,
                                  const DistMetaTensor& output,
                                  const std::vector<int64_t>& target_shape) {
  auto [x_dist_attr, y_dist_attr] = AlignExpandAsDistAttrs(x, output);
  return {{x_dist_attr, y_dist_attr}, {y_dist_attr}};
}

SpmdInfo ExpandAsGradInferSpmd(const DistMetaTensor& x,
                               const DistMetaTensor& out_grad,
                               const std::vector<int64_t>& target_shape) {
  auto [x_dist_attr, y_dist_attr] = AlignExpandAsDistAttrs(x, out_grad);
  const auto& x_dims_mapping = x_dist_attr.dims_mapping();
  const auto& y_dims_mapping = y_dist_attr.dims_mapping();

  // handle partial grad
  auto x_grad_dist_attr = x_dist_attr;
  int x_ndims = x_dims_mapping.size();
  int y_ndims = y_dims_mapping.size();
  int dims_diff = y_ndims - x_ndims;
  std::vector<int64_t> partial;
  for (int i = 0; i < y_ndims; ++i) {
    if (i < dims_diff || x_dims_mapping[i - dims_diff] != y_dims_mapping[i]) {
      if (y_dims_mapping[i] >= 0) {
        partial.push_back(y_dims_mapping[i]);
      }
    }
  }
  x_grad_dist_attr.set_partial_status(partial);
  return {{x_dist_attr, y_dist_attr}, {x_grad_dist_attr}};
}

}  // namespace phi::distributed
