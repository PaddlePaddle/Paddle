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

#include "paddle/phi/infermeta/spmd_rules/roll.h"
#include "glog/logging.h"
#include "paddle/common/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

SpmdInfo RollInferSpmd(const DistMetaTensor& x,
                       const std::vector<int64_t>& shifts,
                       const std::vector<int64_t>& axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  std::vector<int64_t> x_dims_mapping_dst = x_dims_mapping_src;
  if (axis.empty()) {
    PADDLE_ENFORCE_EQ(
        shifts.size(),
        1,
        common::errors::InvalidArgument("When dims.size() == 0, shifts.size() "
                                        "should be equal to 1, But received "
                                        "shifts.size() = %d",
                                        shifts.size()));
    for (int i = 0; i < x_ndim; i++) {
      x_dims_mapping_dst[i] = -1;
    }

  } else {
    PADDLE_ENFORCE_EQ(
        axis.size(),
        shifts.size(),
        common::errors::InvalidArgument("When dims.size() != 0, dims.size() "
                                        "should be equal to "
                                        "shifts.size(). But received "
                                        "dims.size() = %d, shifts.size() = %d",
                                        axis.size(),
                                        shifts.size()));
    for (const auto& i : axis) {
      int64_t axis_i = i < 0 ? x_ndim + i : i;
      PADDLE_ENFORCE_EQ(
          0 <= axis_i && axis_i < x_ndim,
          true,
          phi::errors::InvalidArgument("The axis of roll should "
                                       "be in range [0, %d), but got %d.",
                                       x_ndim,
                                       axis_i));
      x_dims_mapping_dst[axis_i] = -1;
    }
  }
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr out_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  out_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  VLOG(4) << "RollInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_OUTPUT(out_dist_attr_dst);

  return {{x_dist_attr_dst}, {out_dist_attr_dst}};
}

SpmdInfo RollGradInferSpmd(const DistMetaTensor& x,
                           const DistMetaTensor& out_grad,
                           const std::vector<int64_t>& shifts,
                           const std::vector<int64_t>& axis) {
  EXTRACT_SHAPE_AND_DIST_ATTR(x);
  EXTRACT_SHAPE_AND_DIST_ATTR(out_grad);
  PADDLE_ENFORCE_EQ(x_ndim,
                    out_grad_ndim,
                    common::errors::InvalidArgument(
                        "RollGrad: The rank of X [%d] and OutGrad [%d] "
                        "must be the same.",
                        x_ndim,
                        out_grad_ndim));

  // Build einsum notation
  std::string alphabet = "abcdefghijlopqrstuvwxyz";
  std::string x_axes = alphabet.substr(0, x_ndim);
  std::string out_grad_axes = x_axes;

  std::vector<int64_t> x_dims_mapping(x_dims_mapping_src);
  std::vector<int64_t> out_grad_dims_mapping(out_grad_dims_mapping_src);
  if (axis.empty()) {
    PADDLE_ENFORCE_EQ(
        shifts.size(),
        1,
        common::errors::InvalidArgument("When dims.size() == 0, shifts.size() "
                                        "should be equal to 1, But received "
                                        "shifts.size() = %d",
                                        shifts.size()));
    for (int i = 0; i < x_ndim; ++i) {
      x_dims_mapping[i] = -1;
      out_grad_dims_mapping[i] = -1;
    }
  } else {
    PADDLE_ENFORCE_EQ(
        axis.size(),
        shifts.size(),
        common::errors::InvalidArgument("When dims.size() != 0, dims.size() "
                                        "should be equal to "
                                        "shifts.size(). But received "
                                        "dims.size() = %d, shifts.size() = %d",
                                        axis.size(),
                                        shifts.size()));
    for (const auto& i : axis) {
      int64_t axis_i = i < 0 ? x_ndim + i : i;
      PADDLE_ENFORCE_EQ(
          0 <= axis_i && axis_i < x_ndim,
          true,
          phi::errors::InvalidArgument("The axis of roll should "
                                       "be in range [0, %d), but got %d.",
                                       x_ndim,
                                       axis_i));
      x_dims_mapping[axis_i] = -1;
      out_grad_dims_mapping[axis_i] = -1;
    }
  }
  auto axis_to_dim_map = ShardingMergeForTensors(
      {{x_axes, x_dims_mapping}, {out_grad_axes, out_grad_dims_mapping}});
  std::vector<int64_t> x_dims_mapping_dst =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  std::vector<int64_t> out_grad_dims_mapping_dst = x_dims_mapping_dst;
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  TensorDistAttr x_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  out_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);
  x_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping_dst);

  VLOG(4) << "RollGradInferSpmd: Done.";
  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr_dst);

  return {{x_dist_attr_dst, out_grad_dist_attr_dst}, {x_grad_dist_attr_dst}};
}

SpmdInfo RollInferSpmdDynamic(const DistMetaTensor& x,
                              const IntArray& shifts,
                              const std::vector<int64_t>& axis) {
  return RollInferSpmd(x, shifts.GetData(), axis);
}

SpmdInfo RollGradInferSpmdDynamic(const DistMetaTensor& x,
                                  const DistMetaTensor& out_grad,
                                  const IntArray& shifts,
                                  const std::vector<int64_t>& axis) {
  return RollGradInferSpmd(x, out_grad, shifts.GetData(), axis);
}

}  // namespace distributed
}  // namespace phi
