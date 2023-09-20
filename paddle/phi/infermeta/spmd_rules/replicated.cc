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

#include "paddle/phi/infermeta/spmd_rules/replicated.h"

#include "glog/logging.h"

#include "paddle/phi/core/distributed/auto_parallel/dist_attr.h"
#include "paddle/phi/core/distributed/auto_parallel/inferspmd_utils.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace phi {
namespace distributed {

using phi::distributed::auto_parallel::str_join;

////////////////// Utils Functions //////////////////
std::vector<int64_t> GetReplicatedDimsmapping(const int ndim) {
  std::vector<int64_t> dims_mapping(ndim, -1);
  return dims_mapping;
}

////////////////// InferMeta(Contains SPMD) Functions //////////////////
SpmdInfo ReplicatedInferSpmd(const std::vector<const DistMetaTensor*>& ins,
                             const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = ins.size();
  int64_t noutputs = outs.size();

  // Step2: Unshard Output's Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << outs[i]->dist_attr().to_string();
    VLOG(4) << outs[i]->dims().to_str();
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[0]->dist_attr());
    std::vector<int64_t> dst_dims_maping = GetReplicatedDimsmapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_maping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step3: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    int ndim = ins[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_maping = GetReplicatedDimsmapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_maping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "ReplicatedSpmd InferForward:";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(phi::vectorize(ins[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(ins[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(phi::vectorize(outs[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  return {dst_input_dist_attrs, output_dist_attrs};
}

SpmdInfo ReplicatedInferSpmdReverse(
    const std::vector<const DistMetaTensor*>& ins,
    const std::vector<const DistMetaTensor*>& outs) {
  // step1: Build Einsum Notation for input tensor's batch axis
  int64_t ninputs = ins.size();
  int64_t noutputs = outs.size();

  // Step2: Unshard Output's Dims Mapping.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutputs; i++) {
    int ndim = outs[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(outs[i]->dist_attr());
    std::vector<int64_t> dst_dims_maping = GetReplicatedDimsmapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_maping);
    output_dist_attrs.emplace_back(dist_attr_dst);
  }

  // Step3: Merge and get Inputs' Batch Axis New Dims Mapping.
  std::vector<TensorDistAttr> dst_input_dist_attrs;
  for (int64_t i = 0; i < ninputs; i++) {
    int ndim = ins[i]->dims().size();
    TensorDistAttr dist_attr_dst =
        CopyTensorDistAttrForOutput(ins[i]->dist_attr());
    std::vector<int64_t> dst_dims_maping = GetReplicatedDimsmapping(ndim);
    dist_attr_dst.set_dims_mapping(dst_dims_maping);
    dst_input_dist_attrs.emplace_back(dist_attr_dst);
  }

  VLOG(4) << "ReplicatedSpmd InferBackward:";
  for (int64_t i = 0; i < noutputs; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " shape: ["
            << str_join(phi::vectorize(outs[i]->dims())) << "] "
            << "src_dims_mapping: ["
            << str_join(outs[i]->dist_attr().dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(output_dist_attrs[i].dims_mapping()) << "]";
  }

  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(phi::vectorize(ins[i]->dims())) << "] "
            << "dst_dims_mapping: ["
            << str_join(dst_input_dist_attrs[i].dims_mapping()) << "]";
  }

  return {dst_input_dist_attrs, output_dist_attrs};
}

}  // namespace distributed
}  // namespace phi
