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

#include "paddle/phi/infermeta/spmd_rules/where.h"

#include "glog/logging.h"

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi::distributed {
using phi::distributed::auto_parallel::str_join;

SpmdInfo WhereInferSpmd(const DistMetaTensor& condition,
                        const DistMetaTensor& x,
                        const DistMetaTensor& y) {
  auto cond_shape = common::vectorize(condition.dims());
  int cond_ndim = cond_shape.size();
  const auto& cond_dist_attr_src = condition.dist_attr();
  std::vector<int64_t> cond_dims_mapping = cond_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(cond_ndim,
                    cond_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        cond_ndim,
                        cond_dims_mapping.size()));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string cond_axes = alphabet.substr(0, cond_ndim);

  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(x_ndim,
                    x_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        x_ndim,
                        x_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      x_ndim,
      common::errors::InvalidArgument("The condition's rank [%d] and x's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      x_ndim));

  std::string x_axes = alphabet.substr(cond_ndim - x_ndim, x_ndim);

  auto y_shape = common::vectorize(y.dims());
  int y_ndim = y_shape.size();
  const auto& y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(y_ndim,
                    y_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        y_ndim,
                        y_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      y_ndim,
      common::errors::InvalidArgument("The condition's rank [%d] and y's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      y_ndim));

  std::string y_axes = alphabet.substr(cond_ndim - y_ndim, y_ndim);

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(cond_axes, cond_dims_mapping);
  axes_sharding_info.emplace_back(x_axes, x_dims_mapping);
  axes_sharding_info.emplace_back(y_axes, x_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  cond_dims_mapping = GetDimsMappingForAxes(cond_axes, axis_to_dim_map);
  x_dims_mapping = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  y_dims_mapping = GetDimsMappingForAxes(y_axes, axis_to_dim_map);

  auto cond_dist_attr = CopyTensorDistAttrForOutput(cond_dist_attr_src);
  cond_dist_attr.set_dims_mapping(cond_dims_mapping);
  auto x_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr.set_dims_mapping(x_dims_mapping);
  auto y_dist_attr = CopyTensorDistAttrForOutput(y_dist_attr_src);
  y_dist_attr.set_dims_mapping(y_dims_mapping);
  auto out_dist_attr = CopyTensorDistAttrForOutput(cond_dist_attr_src);
  out_dist_attr.set_dims_mapping(cond_dims_mapping);

  VLOG(4) << "WhereInferSpmd:";
  VLOG(4) << "Einsum Notation: " << cond_axes << "," << x_axes << "," << y_axes
          << "-->" << cond_axes;
  VLOG(4) << "cond shape: [" << str_join(cond_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(cond_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(cond_dims_mapping) << "]";

  VLOG(4) << "x shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";

  VLOG(4) << "y shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(y_dims_mapping) << "]";

  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(cond_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return SpmdInfo({cond_dist_attr, x_dist_attr, y_dist_attr}, {out_dist_attr});
}

SpmdInfo WhereInferSpmdReverse(const DistMetaTensor& condition,
                               const DistMetaTensor& x,
                               const DistMetaTensor& y,
                               const DistMetaTensor& output) {
  auto cond_shape = common::vectorize(condition.dims());
  int cond_ndim = cond_shape.size();
  const auto& cond_dist_attr_src = condition.dist_attr();
  std::vector<int64_t> cond_dims_mapping = cond_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(cond_ndim,
                    cond_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        cond_ndim,
                        cond_dims_mapping.size()));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string cond_axes = alphabet.substr(0, cond_ndim);

  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      x_ndim,
      common::errors::InvalidArgument("The x's rank [%d] and x's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      x_ndim));

  std::string x_axes = alphabet.substr(cond_ndim - x_ndim, x_ndim);

  auto y_shape = common::vectorize(y.dims());
  int y_ndim = y_shape.size();
  const auto& y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      y_ndim,
      y_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor y's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      y_ndim,
                                      y_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      y_ndim,
      common::errors::InvalidArgument("The y's rank [%d] and y's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      y_ndim));

  std::string y_axes = alphabet.substr(cond_ndim - y_ndim, y_ndim);

  auto out_shape = common::vectorize(output.dims());
  int out_ndim = out_shape.size();
  const auto& out_dist_attr_src = output.dist_attr();
  const std::vector<int64_t>& out_dims_mapping =
      out_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(out_ndim,
                    out_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor output's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        cond_ndim,
                        cond_dims_mapping.size()));

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{cond_axes, out_dist_attr_src.dims_mapping()}});

  cond_dims_mapping = GetDimsMappingForAxes(cond_axes, axis_to_dim_map);
  x_dims_mapping = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  y_dims_mapping = GetDimsMappingForAxes(y_axes, axis_to_dim_map);

  auto cond_dist_attr = CopyTensorDistAttrForOutput(cond_dist_attr_src);
  cond_dist_attr.set_dims_mapping(cond_dims_mapping);
  auto x_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr.set_dims_mapping(x_dims_mapping);
  auto y_dist_attr = CopyTensorDistAttrForOutput(y_dist_attr_src);
  y_dist_attr.set_dims_mapping(y_dims_mapping);
  auto out_dist_attr = CopyTensorDistAttrForOutput(cond_dist_attr_src);
  out_dist_attr.set_dims_mapping(cond_dims_mapping);

  VLOG(4) << "WhereInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << cond_axes << "," << x_axes << "," << y_axes
          << "-->" << cond_axes;
  VLOG(4) << "cond shape: [" << str_join(cond_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(cond_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(cond_dims_mapping) << "]";

  VLOG(4) << "x shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";

  VLOG(4) << "y shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(y_dims_mapping) << "]";

  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(cond_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return SpmdInfo({cond_dist_attr, x_dist_attr, y_dist_attr}, {out_dist_attr});
}

SpmdInfo WhereGradInferSpmd(const DistMetaTensor& condition,
                            const DistMetaTensor& x,
                            const DistMetaTensor& y,
                            const DistMetaTensor& out_grad) {
  auto cond_shape = common::vectorize(condition.dims());
  int cond_ndim = cond_shape.size();
  const auto& cond_dist_attr_src = condition.dist_attr();
  std::vector<int64_t> cond_dims_mapping = cond_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(cond_ndim,
                    cond_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor Input's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        cond_ndim,
                        cond_dims_mapping.size()));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string cond_axes = alphabet.substr(0, cond_ndim);

  auto x_shape = common::vectorize(x.dims());
  int x_ndim = x_shape.size();
  const auto& x_dist_attr_src = x.dist_attr();
  std::vector<int64_t> x_dims_mapping = x_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      x_ndim,
      x_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor x's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      x_ndim,
                                      x_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      x_ndim,
      common::errors::InvalidArgument("The x's rank [%d] and x's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      x_ndim));

  std::string x_axes = alphabet.substr(cond_ndim - x_ndim, x_ndim);

  auto y_shape = common::vectorize(y.dims());
  int y_ndim = y_shape.size();
  const auto& y_dist_attr_src = y.dist_attr();
  std::vector<int64_t> y_dims_mapping = y_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      y_ndim,
      y_dims_mapping.size(),
      common::errors::InvalidArgument("The Tensor y's rank [%d] and Input's "
                                      "dims_mapping size [%d] are not matched.",
                                      y_ndim,
                                      y_dims_mapping.size()));

  PADDLE_ENFORCE_GE(
      cond_ndim,
      y_ndim,
      common::errors::InvalidArgument("The y's rank [%d] and y's "
                                      "rank [%d] are not matched.",
                                      cond_ndim,
                                      y_ndim));

  std::string y_axes = alphabet.substr(cond_ndim - y_ndim, y_ndim);

  auto out_grad_shape = common::vectorize(out_grad.dims());
  int out_grad_ndim = out_grad_shape.size();
  const auto& out_grad_dist_attr_src = out_grad.dist_attr();
  std::vector<int64_t> out_grad_dims_mapping =
      out_grad_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(out_grad_ndim,
                    out_grad_dims_mapping.size(),
                    common::errors::InvalidArgument(
                        "The Tensor output's rank [%d] and Input's "
                        "dims_mapping size [%d] are not matched.",
                        out_grad_ndim,
                        out_grad_dims_mapping.size()));

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info.emplace_back(cond_axes, cond_dims_mapping);
  axes_sharding_info.emplace_back(x_axes, x_dims_mapping);
  axes_sharding_info.emplace_back(y_axes, x_dims_mapping);
  axes_sharding_info.emplace_back(cond_axes, out_grad_dims_mapping);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  cond_dims_mapping = GetDimsMappingForAxes(cond_axes, axis_to_dim_map);
  x_dims_mapping = GetDimsMappingForAxes(x_axes, axis_to_dim_map);
  y_dims_mapping = GetDimsMappingForAxes(y_axes, axis_to_dim_map);

  auto cond_dist_attr = CopyTensorDistAttrForOutput(cond_dist_attr_src);
  cond_dist_attr.set_dims_mapping(cond_dims_mapping);
  auto x_dist_attr = CopyTensorDistAttrForOutput(x_dist_attr_src);
  x_dist_attr.set_dims_mapping(x_dims_mapping);
  auto y_dist_attr = CopyTensorDistAttrForOutput(y_dist_attr_src);
  y_dist_attr.set_dims_mapping(y_dims_mapping);
  auto out_grad_dist_attr = CopyTensorDistAttrForOutput(out_grad_dist_attr_src);
  out_grad_dist_attr.set_dims_mapping(cond_dims_mapping);

  std::vector<int64_t> x_partial_on_dims;
  const auto& dim_mapping = cond_dims_mapping;
  for (int i = 0; i < cond_ndim - x_ndim; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      x_partial_on_dims.push_back(mapping);
    }
  }

  std::vector<int64_t> y_partial_on_dims;
  for (int i = 0; i < cond_ndim - y_ndim; ++i) {
    auto mapping = dim_mapping[i];
    if (mapping != -1) {
      y_partial_on_dims.push_back(mapping);
    }
  }
  auto x_grad = CopyTensorDistAttrForOutput(x_dist_attr);
  x_grad.set_dims_mapping(x_dims_mapping);
  x_grad.set_partial_status(x_partial_on_dims);
  auto y_grad = CopyTensorDistAttrForOutput(y_dist_attr);
  y_grad.set_dims_mapping(y_dims_mapping);
  y_grad.set_partial_status(y_partial_on_dims);

  VLOG(4) << "WhereInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << cond_axes << "," << x_axes << "," << y_axes
          << "-->" << cond_axes;
  VLOG(4) << "cond shape: [" << str_join(cond_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(cond_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(cond_dims_mapping) << "]";

  VLOG(4) << "x shape: [" << str_join(x_shape) << "] "
          << "src_dims_mapping: [" << str_join(x_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(x_dims_mapping) << "]";

  VLOG(4) << "y shape: [" << str_join(y_shape) << "] "
          << "src_dims_mapping: [" << str_join(y_dist_attr_src.dims_mapping())
          << "] "
          << "dst_dims_mapping: [" << str_join(y_dims_mapping) << "]";

  VLOG(4) << "out_grad shape: [" << str_join(out_grad_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(out_grad_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_grad_dims_mapping) << "]";

  VLOG(4) << "x_grad"
          << " dims_mapping: [" << str_join(x_dims_mapping) << "]";

  VLOG(4) << "y_grad"
          << " dims_mapping: [" << str_join(y_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return SpmdInfo(
      {cond_dist_attr, x_dist_attr, y_dist_attr, out_grad_dist_attr},
      {x_grad, y_grad});
}
}  // namespace phi::distributed
