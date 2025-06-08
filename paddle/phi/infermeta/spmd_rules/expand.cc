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

#include "paddle/phi/infermeta/spmd_rules/expand.h"

#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

SpmdInfo ExpandInferSpmd(const DistMetaTensor& x, const IntArray& shape) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  auto expand_shape = shape.GetData();
  std::vector<int64_t> out_dims_mapping(shape.size());
  int diff = expand_shape.size() - x_shape.size();
  for (int i = expand_shape.size() - 1; i >= diff; --i) {
    if (expand_shape[i] != -1 && expand_shape[i] != x_shape[i - diff]) {
      out_dims_mapping[i] = -1;
    } else {
      out_dims_mapping[i] = x_dims_mapping_src[i - diff];
    }
  }
  for (int i = 0; i < diff; i++) {
    out_dims_mapping[i] = -1;
  }
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  return {{x_dist_attr_src}, {out_dist_attr}};
}

SpmdInfo ExpandGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& out_grad,
                             const IntArray& shape) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  size_t axis =
      std::abs(static_cast<int>(out_grad.dims().size() - x.dims().size()));
  std::vector<int64_t> x_grad_dims_mapping(x_ndim, -1);
  std::vector<int64_t> partial_dims;
  for (size_t i = 0; i < out_grad_dims_mapping_src.size(); ++i) {
    if (i < axis || i >= axis + x.dims().size() ||
        out_grad.dims()[i] != x.dims()[i - axis]) {
      if (out_grad_dims_mapping_src[i] >= 0) {
        partial_dims.push_back(out_grad_dims_mapping_src[i]);
      }
      continue;
    }
    x_grad_dims_mapping[i - axis] = out_grad_dims_mapping_src[i];
  }
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr x_dist_attr = x_grad_dist_attr;
  x_dist_attr.set_dims_mapping(x_grad_dims_mapping);
  x_grad_dist_attr.set_dims_mapping(x_grad_dims_mapping);
  x_grad_dist_attr.set_partial_status(partial_dims);
  return {{x_dist_attr, out_grad_dist_attr_src}, {x_grad_dist_attr}};
}

}  // namespace phi::distributed
