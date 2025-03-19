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

#include "paddle/phi/infermeta/spmd_rules/default_data_parallel.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi::distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::vector<int64_t> GetDefaultDataParallelDimsMapping(
    const int64_t batch_axis_dim, const int ndim) {
  std::vector<int64_t> dims_mapping(ndim, -1);
  dims_mapping[0] = batch_axis_dim;
  return dims_mapping;
}

////////////////// InferMeta(Contains SPMD) Functions //////////////////

SpmdInfo DefaultDataParallelInferSpmd(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = static_cast<int64_t>(ins.size());
  int64_t noutputs = static_cast<int64_t>(outs.size());
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  std::string batch_axis = "b";

  for (int64_t i = 0; i < ninputs; ++i) {
    axes_sharding_info.push_back(
        {batch_axis, {ins[i]->dist_attr().dims_mapping()[0]}});
  }

  // Step2: Sharding Merge
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);
  int64_t batch_axis_dim = axis_to_dim_map[batch_axis];

  // Step3: Infer Output's Batch Axis Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[0]->dist_attr());
    std::vector<int64_t> dst_dims_mapping =
        GetDefaultDataParallelDimsMapping(batch_axis_dim, ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step4: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    int ndim = ins[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping =
        GetDefaultDataParallelDimsMapping(batch_axis_dim, ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "DefaultDataParallelSpmd InferForward:";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(ins[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(ins[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(outs[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  return {ToArgDistAttr(dst_input_dist_attrs),
          ToArgDistAttr(output_dist_attrs)};
}
SpmdInfo DefaultDataParallelInferSpmdReverse(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = static_cast<int64_t>(ins.size());
  int64_t noutputs = static_cast<int64_t>(outs.size());
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  std::string batch_axis = "b";

  for (int64_t i = 0; i < noutputs; ++i) {
    axes_sharding_info.push_back(
        {batch_axis, {outs[i]->dist_attr().dims_mapping()[0]}});
  }

  // Step2: Sharding Merge
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);
  int64_t batch_axis_dim = axis_to_dim_map[batch_axis];

  // Step3: Infer Output's Batch Axis Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(outs[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping =
        GetDefaultDataParallelDimsMapping(batch_axis_dim, ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step4: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    int ndim = ins[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_mapping =
        GetDefaultDataParallelDimsMapping(batch_axis_dim, ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_mapping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "DefaultDataParallelSpmd InferBackward:";
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(outs[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(outs[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(common::vectorize(ins[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  return {ToArgDistAttr(dst_input_dist_attrs),
          ToArgDistAttr(output_dist_attrs)};
}

}  // namespace phi::distributed
