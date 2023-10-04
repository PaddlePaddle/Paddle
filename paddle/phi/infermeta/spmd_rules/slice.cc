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

#include "paddle/phi/infermeta/spmd_rules/slice.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"
#include "paddle/phi/infermeta/spmd_rules/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

SpmdInfo SliceInferSpmd(const DistMetaTensor& input,
                        const std::vector<int64_t>& axes,
                        const std::vector<int>& starts,
                        const std::vector<int>& ends,
                        const std::vector<int64_t>& infer_flags,
                        const std::vector<int64_t>& decrease_axis) {
  auto input_shape = phi::vectorize(input.dims());
  int input_ndim = input_shape.size();
  auto input_dist_attr_src = input.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr_src.dims_mapping();
  PADDLE_ENFORCE_EQ(
      input_ndim,
      input_dims_mapping.size(),
      phi::errors::InvalidArgument("The Tensor Input's rank [%d] and Input's "
                                   "dims_mapping size [%d] are not matched.",
                                   input_ndim,
                                   input_dims_mapping.size()));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string input_axes = alphabet.substr(0, input_ndim);
  std::string special_axes = alphabet.substr(input_ndim);

  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    input_axes[axis] = special_axes[i];
  }

  std::string out_axes(input_axes);

  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    out_axes[axis] = '1';
  }

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{input_axes, input_dims_mapping}});

  std::vector<int64_t> out_dims_mapping =
      GetDimsMappingForAxes(out_axes, axis_to_dim_map);

  TensorDistAttr out_dist_attr =
      CopyTensorDistAttrForOutput(input_dist_attr_src);
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  TensorDistAttr input_dist_attr_dst(input_dist_attr_src);
  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    input_dims_mapping[axis] = -1;
  }
  input_dist_attr_dst.set_dims_mapping(input_dims_mapping);

  VLOG(4) << "SliceInferSpmd:";
  VLOG(4) << "Einsum Notation: " << input_axes << "-->" << out_axes;
  VLOG(4) << "Input shape: [" << str_join(input_shape) << "] "
          << "src_dims_mapping: ["
          << str_join(input_dist_attr_src.dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(input_dims_mapping) << "]";
  VLOG(4) << "Output"
          << " dims_mapping: [" << str_join(out_dims_mapping) << "]";
  VLOG(4) << std::endl;

  return {{input_dist_attr_dst}, {out_dist_attr}};
}

SpmdInfo SliceInferSpmdReverse(const DistMetaTensor& input,
                               const DistMetaTensor& output,
                               const std::vector<int64_t>& axes,
                               const std::vector<int>& starts,
                               const std::vector<int>& ends,
                               const std::vector<int64_t>& infer_flags,
                               const std::vector<int64_t>& decrease_axis) {
  auto output_shape = phi::vectorize(output.dims());
  int out_ndim = output_shape.size();
  auto out_dist_attr = output.dist_attr();
  int out_dims_mapping_size = out_dist_attr.dims_mapping().size();
  auto input_shape = phi::vectorize(input.dims());
  int input_ndim = input_shape.size();
  auto input_dist_attr = input.dist_attr();
  std::vector<int64_t> input_dims_mapping = input_dist_attr.dims_mapping();

  PADDLE_ENFORCE_EQ(
      input_ndim,
      out_ndim,
      phi::errors::InvalidArgument("The Tensor Input's rank [%d] is not equal "
                                   "to the Tensor Output's rank [%d]",
                                   input_ndim,
                                   out_ndim));

  PADDLE_ENFORCE_EQ(
      out_ndim,
      out_dims_mapping_size,
      phi::errors::InvalidArgument("The Tensor Output's rank [%d] and Its "
                                   "dims_mapping size [%d] are not matched.",
                                   out_ndim,
                                   out_dims_mapping_size));

  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::string input_axes = alphabet.substr(0, input_ndim);
  std::string special_axes = alphabet.substr(input_ndim);

  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    input_axes[axis] = special_axes[i];
  }

  std::string out_axes(input_axes);

  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    out_axes[axis] = special_axes[i];
  }

  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  std::vector<int64_t> out_dims_mapping = output.dist_attr().dims_mapping();
  axes_sharding_info.emplace_back(std::make_pair(out_axes, out_dims_mapping));

  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  input_dims_mapping = GetDimsMappingForAxes(input_axes, axis_to_dim_map, true);
  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    input_dims_mapping[axis] = -1;
  }
  input_dist_attr.set_dims_mapping(input_dims_mapping);
  out_dims_mapping = GetDimsMappingForAxes(out_axes, axis_to_dim_map, true);
  for (int i = 0; i < static_cast<int>(axes.size()); i++) {
    int axis = axes[i] < 0 ? axes[i] + input_ndim : axes[i];
    out_dims_mapping[axis] = -1;
  }
  out_dist_attr.set_dims_mapping(out_dims_mapping);

  VLOG(4) << "SliceInferSpmdReverse:";
  VLOG(4) << "Einsum Notation: " << input_axes << "-->" << out_axes;
  VLOG(4) << "Output"
          << " shape: [" << str_join(phi::vectorize(output.dims())) << "] "
          << "src_dims_mapping: ["
          << str_join(output.dist_attr().dims_mapping()) << "] "
          << "dst_dims_mapping: [" << str_join(out_dist_attr.dims_mapping())
          << "]";
  VLOG(4) << "Input shape: [" << str_join(input_shape) << "] "
          << "dims_mapping: [" << str_join(input_dims_mapping) << "]\n\n";

  return {{input_dist_attr}, {out_dist_attr}};
}

}  // namespace distributed
}  // namespace phi
