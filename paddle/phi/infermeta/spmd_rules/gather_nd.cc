/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/infermeta/spmd_rules/gather_nd.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo GatherNdInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& index) {
  // Step0: Verify Input Args Based on Gather Logic
  // extract and check x_ndim, x_shape, x_dist_attr_src and
  // x_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);

  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  std::vector<int64_t> index_dims_mapping(index_dims_mapping_src);

  int index_axis = index_shape[index_ndim - 1];
  index_dims_mapping[index_ndim - 1] = -1;

  for (int axis = 0; axis < index_axis; axis++) {
    x_dims_mapping[axis] = -1;
  }

  std::vector<int64_t> out_dims_mapping;
  for (int i = 0; i < index_ndim - 1; ++i) {
    out_dims_mapping.emplace_back(index_dims_mapping[i]);
  }
  for (int i = index_axis; i < x_ndim; ++i) {
    out_dims_mapping.emplace_back(x_dims_mapping[i]);
  }

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);

  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  std::vector<int64_t> out_shape;
  for (int i = 0; i < index_ndim - 1; ++i) {
    out_shape.emplace_back(index_shape[i]);
  }
  for (int i = static_cast<int>(index_shape[index_ndim - 1]); i < x_ndim; ++i) {
    out_shape.emplace_back(x_shape[i]);
  }

  TensorDistAttr out_dist_attr = TensorDistAttr(out_shape);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  VLOG(4) << "out";
  VLOG(4) << "dist_attr: [" << out_dist_attr.to_string() << "]";
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo GatherNdInferSpmdReverse(const DistMetaTensor& x,
                                  const DistMetaTensor& index,
                                  const DistMetaTensor& out) {
  // Step0: Verify Input Args Based on Gather Logic
  // extract and check out_ndim, out_shape, out_dist_attr_src and
  // out_dims_mapping_src with the macro
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(index);
  EXTRACT_SHAPE_AND_DIST_ATTR(out);

  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  std::vector<int64_t> index_dims_mapping(index_dims_mapping_src);
  std::vector<int64_t> out_dims_mapping(out_dims_mapping_src);

  for (int axis = 0; axis < index_ndim - 1; ++axis) {
    index_dims_mapping[axis] = out_dims_mapping[axis];
  }
  index_dims_mapping[index_ndim - 1] = -1;

  int index_axis = index_shape[index_ndim - 1];
  for (int axis = 0; axis < index_axis; axis++) {
    x_dims_mapping[axis] = -1;
  }
  for (int axis = x_ndim - 1; axis >= index_axis; axis--) {
    x_dims_mapping[axis] = out_dims_mapping[out_ndim + (axis - x_ndim)];
  }

  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr index_dist_attr_dst =
      CopyTensorDistAttrForOutput(index_dist_attr_src);
  index_dist_attr_dst.set_dims_mapping(index_dims_mapping);

  VLOG(4) << "out dist_attr: [" << out_dist_attr_src.to_string() << "]";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(index);
  VLOG(4) << std::endl;
  return {{x_dist_attr_dst, index_dist_attr_dst}, {out_dist_attr_src}};
}

}  // namespace phi::distributed
