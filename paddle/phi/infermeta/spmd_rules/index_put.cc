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

#include "paddle/phi/infermeta/spmd_rules/index_put.h"

#include "glog/logging.h"
#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
SpmdInfo IndexPutInferSpmd(const DistMetaTensor& x,
                           const std::vector<DistMetaTensor>& indices,
                           const DistMetaTensor& value,
                           const bool accumulate) {
  // Step0: verify input args based on group_norm logic
  auto x_shape = common::vectorize(x.dims());
  int indices_size = indices.size();
  auto indices_shape = common::vectorize(indices[0].dims());
  auto value_shape = common::vectorize(value.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int indices_ndim = static_cast<int>(indices_shape.size());
  int value_ndim = static_cast<int>(value_shape.size());

  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<TensorDistAttr> indices_dist_attrs_src;
  std::transform(indices.begin(),
                 indices.end(),
                 std::back_inserter(indices_dist_attrs_src),
                 [](auto& meta) { return meta.dist_attr(); });
  TensorDistAttr value_dist_attr_src = value.dist_attr();

  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();

  PADDLE_ENFORCE_GE(x_ndim,
                    indices_size,
                    common::errors::InvalidArgument(
                        "The ndim of x in index_put should be "
                        "greater than or equal to the size of indices, "
                        "but got x_ndim:[%d],indices_size:[%d].",
                        x_ndim,
                        indices_size));

  PADDLE_ENFORCE_LE(
      value_ndim,
      x_ndim - indices_size + 1,
      common::errors::InvalidArgument("The ndim of value in index_put should "
                                      "be less than or equal to [%d], "
                                      "but got value_ndim:[%d].",
                                      x_ndim - indices_size + 1,
                                      value_ndim));
  PADDLE_ENFORCE_EQ(
      indices_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of indices in index_put should be equal to 1, "
          "but got indices_ndim:[%d].",
          indices_ndim));
  for (int i = 0; i < indices_size; i++) {
    PADDLE_ENFORCE_EQ(
        indices[i].dims().size(),
        1,
        common::errors::InvalidArgument(
            "The ndim of indices[%d] in index_put should be equal to 1, "
            "but got indices[%d] ndim:[%d].",
            i,
            i,
            indices[i].dims().size()));
  }
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  std::string value_axes(value_ndim, '1');
  int index = indices_size - 1;
  for (int i = 0; i < value_ndim; ++i) {
    value_axes[i] = x_axes[index++];
  }

  // Step1: set dims_mapping for input
  for (int i = 0; i < indices_size; i++) {
    x_dims_mapping[i] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});
  // Step2: set dims_mapping for output
  TensorDistAttr out_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_dist_attr.set_dims_mapping(x_dims_mapping);
  // Step3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr value_dist_attr_dst =
      CopyTensorDistAttrForOutput(value.dist_attr());
  value_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(value_axes, axis_to_dim_map));
  std::vector<TensorDistAttr> indices_dist_attrs_dst = indices_dist_attrs_src;
  for (auto& input_attr : indices_dist_attrs_dst) {
    input_attr.set_dims_mapping(std::vector<int64_t>{-1});
  }
  // Step4: Log SpmdInfo
  LOG_SPMD_INPUT(x);
  // LOG_SPMD_INPUT(indices);
  VLOG(4) << "name: indices";
  VLOG(4) << "ndim: " << std::to_string(indices_ndim) << " "
          << "indices_size: " << std::to_string(indices_size) << " "
          << "indices_dist_attr_src: [" << indices_dist_attrs_src[0].to_string()
          << "] "
          << "indices_dist_attr_dst: [" << indices_dist_attrs_dst[0].to_string()
          << "]";

  LOG_SPMD_INPUT(value);
  LOG_SPMD_OUTPUT(out_dist_attr);

  return {{x_dist_attr_dst, indices_dist_attrs_dst, value_dist_attr_dst},
          {out_dist_attr}};
}

SpmdInfo IndexPutGradInferSpmd(const DistMetaTensor& x,
                               const std::vector<DistMetaTensor>& indices,
                               const DistMetaTensor& value,
                               const DistMetaTensor& out_grad,
                               const bool accumulate) {
  // Step0: verify input args based on group_norm logic
  auto x_shape = common::vectorize(x.dims());
  int indices_size = indices.size();
  auto indices_shape = common::vectorize(indices[0].dims());
  auto value_shape = common::vectorize(value.dims());
  auto out_grad_shape = common::vectorize(out_grad.dims());
  int x_ndim = static_cast<int>(x_shape.size());
  int indices_ndim = static_cast<int>(indices_shape.size());
  int value_ndim = static_cast<int>(value_shape.size());
  int out_grad_ndim = static_cast<int>(out_grad_shape.size());
  TensorDistAttr x_dist_attr_src = x.dist_attr();
  std::vector<TensorDistAttr> indices_dist_attrs_src;
  std::transform(indices.begin(),
                 indices.end(),
                 std::back_inserter(indices_dist_attrs_src),
                 [](auto& meta) { return meta.dist_attr(); });
  TensorDistAttr value_dist_attr_src = value.dist_attr();
  TensorDistAttr out_grad_dist_attr_src = out_grad.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      out_grad_ndim,
      x_ndim,
      common::errors::InvalidArgument(
          "The ndim of out_grad in index_put_grad should be equal to the "
          "ndim of x, but got out_grad_ndim:[%d],x_ndim:[%d].",
          out_grad_ndim,
          x_ndim));
  PADDLE_ENFORCE_GE(x_ndim,
                    indices_size,
                    common::errors::InvalidArgument(
                        "The ndim of x in index_put should be "
                        "greater than or equal to the size of indices, "
                        "but got x_ndim:[%d],indices_size:[%d].",
                        x_ndim,
                        indices_size));

  PADDLE_ENFORCE_LE(
      value_ndim,
      x_ndim - indices_size + 1,
      common::errors::InvalidArgument("The ndim of value in index_put should "
                                      "be less than or equal to [%d], "
                                      "but got value_ndim:[%d].",
                                      x_ndim - indices_size + 1,
                                      value_ndim));
  PADDLE_ENFORCE_EQ(
      indices_ndim,
      1,
      common::errors::InvalidArgument(
          "The ndim of indices in index_put should be equal to 1, "
          "but got indices_ndim:[%d].",
          indices_ndim));
  for (int i = 0; i < indices_size; i++) {
    PADDLE_ENFORCE_EQ(
        indices[i].dims().size(),
        1,
        common::errors::InvalidArgument(
            "The ndim of indices[%d] in index_put should be equal to 1, "
            "but got indices[%d] ndim:[%d].",
            i,
            i,
            indices[i].dims().size()));
  }
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  for (int i = 0; i < x_ndim; ++i) {
    x_axes[i] = alphabet[i];
  }
  std::string value_axes(value_ndim, '1');
  int index = indices_size - 1;
  for (int i = 0; i < value_ndim; ++i) {
    value_axes[i] = x_axes[index++];
  }
  // Step1: set x_dims_mapping
  for (int i = 0; i < indices_size; i++) {
    x_dims_mapping[i] = -1;
  }
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});
  // Step2: set dims_mapping for output
  TensorDistAttr x_grad_dist_attr =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_grad_dist_attr.set_dims_mapping(x_dims_mapping);
  TensorDistAttr value_grad_dist_attr =
      CopyTensorDistAttrForOutput(value_dist_attr_src);
  value_grad_dist_attr.set_dims_mapping(
      GetDimsMappingForAxes(value_axes, axis_to_dim_map));
  // Step3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr out_grad_dist_attr_dst =
      CopyTensorDistAttrForOutput(x_dist_attr_src);
  out_grad_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr value_dist_attr_dst =
      CopyTensorDistAttrForOutput(value.dist_attr());
  value_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(value_axes, axis_to_dim_map));
  std::vector<TensorDistAttr> indices_dist_attrs_dst = indices_dist_attrs_src;
  for (auto& input_attr : indices_dist_attrs_dst) {
    input_attr.set_dims_mapping(std::vector<int64_t>{-1});
  }
  // Step4: Log SpmdInfo
  LOG_SPMD_INPUT(x);
  // LOG_SPMD_INPUT(indices);
  VLOG(4) << "name: indices";
  VLOG(4) << "ndim: " << std::to_string(indices_ndim) << " "
          << "indices_size: " << std::to_string(indices_size) << " "
          << "indices_dist_attr_src: [" << indices_dist_attrs_src[0].to_string()
          << "] "
          << "indices_dist_attr_dst: [" << indices_dist_attrs_dst[0].to_string()
          << "]";

  LOG_SPMD_INPUT(value);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad_dist_attr);
  LOG_SPMD_OUTPUT(value_grad_dist_attr);

  return {{x_dist_attr_dst,
           indices_dist_attrs_dst,
           value_dist_attr_dst,
           out_grad_dist_attr_dst},
          {x_grad_dist_attr, value_grad_dist_attr}};
}

}  // namespace phi::distributed
