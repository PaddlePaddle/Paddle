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

#include "paddle/phi/infermeta/spmd_rules/rms_norm.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/spmd_rule_macro_define.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo RmsNormInferSpmd(const DistMetaTensor& x,
                          const DistMetaTensor& scale,
                          float epsilon) {
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scale);

  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_axes(x_ndim, '1');
  std::string variance_axes(x_ndim - 1, '1');
  // allow axis before begin_norm_axis be sharded
  for (int i = 0; i < x_ndim - 1; ++i) {
    x_axes[i] = alphabet[i];
    variance_axes[i] = alphabet[i];
  }
  // x_axes[0] = alphabet[0];
  std::string scale_axes(1, x_axes[x_ndim - 1]);

  // get output notation
  std::string out_axes = x_axes;

  auto x_dims_mapping = x_dims_mapping_src;
  x_dims_mapping[x_ndim - 1] = -1;
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{x_axes, x_dims_mapping}});

  // Step2.2: infer output dims mapping
  TensorDistAttr out = CopyTensorDistAttrForOutput(x_dist_attr_src);
  TensorDistAttr invvar = CopyTensorDistAttrForOutput(x_dist_attr_src);
  out.set_dims_mapping(GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  invvar.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  // Step2.3: update input dims mapping
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  // TODO(zhiqiu): support sharding on scale and bias
  // Now, apply replicating.
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale_dist_attr_src);
  scale_dist_attr_dst.set_dims_mapping({-1});

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);

  LOG_SPMD_OUTPUT(out);
  LOG_SPMD_OUTPUT(invvar);

  return {{x_dist_attr_dst, scale_dist_attr_dst}, {out, invvar}};
}

SpmdInfo RmsNormInferSpmdReverse(const DistMetaTensor& x,
                                 const DistMetaTensor& scale,
                                 const DistMetaTensor& out,
                                 const DistMetaTensor& invvar,
                                 float epsilon) {
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scale);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(out);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(invvar);

  std::string alphabet = "ijklmnopqrstuvwxyz";
  // the axes after norm_axis should be replicated,
  // so set their notation to '1'.
  std::string x_axes(out_ndim, '1');
  std::string variance_axes(out_ndim - 1, '1');
  // allow axis before begin_norm_axis be sharded
  for (int i = 0; i < out_ndim - 1; ++i) {
    x_axes[i] = alphabet[i];
    variance_axes[i] = alphabet[i];
  }
  auto out_axes = x_axes;
  std::string scale_axes(1, x_axes[x_ndim - 1]);

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(out_axes, out_dims_mapping_src);
  axes_sharding_info.emplace_back(variance_axes, invvar_dims_mapping_src);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  auto out_dist_attr_dst = CopyTensorDistAttrForOutput(out_dist_attr_src);
  auto invvar_dist_attr_dst = CopyTensorDistAttrForOutput(invvar_dist_attr_src);
  out_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(out_axes, axis_to_dim_map));
  invvar_dist_attr_dst.set_dims_mapping(
      GetDimsMappingForAxes(variance_axes, axis_to_dim_map));

  // Step2.2 infer input dims mapping
  std::vector<int64_t> x_dims_mapping =
      GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  TensorDistAttr x_dist_attr_dst = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr_dst.set_dims_mapping(x_dims_mapping);
  TensorDistAttr scale_dist_attr_dst =
      CopyTensorDistAttrForOutput(scale_dist_attr_src);
  scale_dist_attr_dst.set_dims_mapping({-1});

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(out);
  LOG_SPMD_INPUT(invvar);
  return {{x_dist_attr_dst, scale_dist_attr_dst},
          {out_dist_attr_dst, invvar_dist_attr_dst}};
}

std::tuple<std::vector<std::string>, std::string> BuildRmsNormGradEinsum(
    int64_t input_rank) {
  std::string alphabet = "ijklmnopqrstuvwxyz";
  std::string x_notation = alphabet.substr(0, input_rank);
  std::string variance_notation = x_notation.substr(0, input_rank - 1);
  std::string align_notation = x_notation.substr(0, input_rank - 1);
  return {{x_notation, variance_notation, x_notation}, align_notation};
}

SpmdInfo RmsNormGradInferSpmd(const DistMetaTensor& x,
                              const DistMetaTensor& scale,
                              const DistMetaTensor& invvar,
                              const DistMetaTensor& out_grad,
                              float epsilon) {
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(x);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(scale);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(invvar);
  EXTRACT_SHAPE_AND_DIST_ATTR_WITH_DIM_CK(out_grad);

  PADDLE_ENFORCE_EQ(
      x_shape.size(),
      out_grad_shape.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Tensor "
                                      "out_grad's rank [%d] are not matched.",
                                      x_shape.size(),
                                      out_grad_shape.size()));

  // 2、align sharding
  std::vector<TensorDistAttr> dist_attrs;
  dist_attrs.push_back(x_dist_attr_src);
  dist_attrs.push_back(invvar_dist_attr_src);
  dist_attrs.push_back(out_grad_dist_attr_src);

  std::vector<std::vector<int64_t>> shapes = {x_shape, invvar_shape, x_shape};
  std::vector<std::string> annotations;
  std::string align_annotation;
  std::tie(annotations, align_annotation) =
      BuildRmsNormGradEinsum(x_shape.size());
  AlignDimsSharding(
      &dist_attrs, shapes, annotations, {}, align_annotation, false);
  auto x_dist_attr_dst = dist_attrs[0];
  auto invvar_dist_attr_dst = dist_attrs[1];
  auto out_grad_dist_attr_dst = dist_attrs[2];

  // TODO(liuzhenhai): support sharded scale and bias
  auto scale_dist_attr_dst = GetReplicatedDistAttr(scale_dist_attr_src);
  auto scale_grad = GetReplicatedDistAttr(scale_dist_attr_src);
  // partial grad dim
  std::vector<int64_t> partial_on_dims;
  const auto& dim_mapping = x_dist_attr_dst.dims_mapping();
  for (size_t i = 0; i + 1 < static_cast<size_t>(x_ndim); ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      partial_on_dims.push_back(mapping);
    }
  }
  scale_grad.set_partial_status(partial_on_dims);
  auto x_grad = out_grad_dist_attr_dst;

  LOG_SPMD_INPUT(x);
  LOG_SPMD_INPUT(scale);
  LOG_SPMD_INPUT(invvar);
  LOG_SPMD_INPUT(out_grad);
  LOG_SPMD_OUTPUT(x_grad);
  LOG_SPMD_OUTPUT(scale_grad);

  return SpmdInfo({x_dist_attr_dst,
                   scale_dist_attr_dst,
                   invvar_dist_attr_dst,
                   out_grad_dist_attr_dst},
                  {x_grad, scale_grad});
}

}  // namespace phi::distributed
