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

#include "paddle/phi/infermeta/spmd_rules/topk.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo TopkInferSpmd(const DistMetaTensor& x,
                       const Scalar& k,
                       int axis,
                       bool largest,
                       bool sorted) {
  // Verify input args
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  axis = axis < 0 ? x_ndim + axis : axis;

  // Infer output dims mapping from merged input dims mapping
  std::vector<int64_t> x_dims_mapping_dst(x_dims_mapping_src);
  std::vector<int64_t> out_dims_mapping;
  std::vector<int64_t> indices_dims_mapping;
  x_dims_mapping_dst[axis] = -1;
  out_dims_mapping.assign(x_dims_mapping_dst.begin(), x_dims_mapping_dst.end());
  indices_dims_mapping = out_dims_mapping;

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);
  TensorDistAttr indices_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  indices_dist_attr.set_dims_mapping(indices_dims_mapping);

  return {{x_dist_attr_dst}, {out_dist_attr, indices_dist_attr}};
}

SpmdInfo TopkGradInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& indices,
                           Tensor out_grad,
                           Scalar k,
                           int axis,
                           bool largest,
                           bool sorted) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(indices);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);

  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  out_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_src);

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_src);

  TensorDistAttr indices_dist_attr_dst =
      CopyTensorDistAttrForOutput(indices_dist_attr_src);
  indices_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_src);

  TensorDistAttr x_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr_dst.set_dims_mapping(out_grad_dims_mapping_src);
  return {{x_dist_attr_dst, indices_dist_attr_dst, out_grad_dist_attr_dst},
          {x_grad_dist_attr_dst}};
}

}  // namespace distributed
}  // namespace phi
