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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/transpose_spmd_rule.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
using phi::distributed::auto_parallel::str_join;
std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
TransposeSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                                const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Transpose Logic
  int64_t ninputs = input_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in transpose must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  VerifySpecs(input_specs, "transpose");

  // step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for input
  int64_t ndim = input_specs[0].shape().size();
  std::vector<std::string> input_axes_vec;
  std::string input_axes = alphabet.substr(0, ndim);
  input_axes_vec.emplace_back(input_axes);

  // get einsum notation for output
  std::string output_axes = GetOutputNotation(ndim, input_axes, attrs);

  // step2: Sharding Propogation
  // step2.1: merge input shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info = GetAxesDimsMappingPair(input_axes_vec, input_specs);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // step2.2: infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> output_dims_mapping =
      GetDimsMappingForAxes(output_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr output_dist_attr =
      CopyTensorDistAttrForOutput(input_specs[0].dist_attr());
  output_dist_attr.set_dims_mapping(output_dims_mapping);

  // step3  Handle partial (TODO)

  VLOG(4) << "TransposeSPMDRule InferForward:";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "src_dims_mapping: [" << str_join(input_specs[i].dims_mapping())
            << "] "
            << "dst_dims_mapping: [" << str_join(input_specs[i].dims_mapping())
            << "]";
  }
  VLOG(4) << "Perm: ["
          << str_join(ExtractAttr<std::vector<int64_t>>("perm", attrs)) << "]";
  VLOG(4) << "Output dims_mapping: [" + str_join(output_dims_mapping) + "]\n\n";

  return {{input_specs[0].dist_attr()}, {output_dist_attr}};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
TransposeSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& input_specs,
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Transpose Logic
  int64_t ninputs = input_specs.size();
  int64_t noutputs = output_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in transpose must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  PADDLE_ENFORCE_EQ(
      noutputs,
      1,
      phi::errors::InvalidArgument("The size of OutputSpec in transpose must "
                                   "be equal to 1, but got [%d].",
                                   noutputs));
  VerifySpecs(output_specs, "transpose_backward");

  // step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  // get einsum notation for input
  int64_t ndim = input_specs[0].shape().size();
  std::string input_axes = alphabet.substr(0, ndim);

  // get einsum notation for output
  std::string output_axes = GetOutputNotation(ndim, input_axes, attrs);
  std::vector<std::string> output_axes_vec;
  output_axes_vec.emplace_back(output_axes);

  // step2: Sharding Propogation
  // step2.1: merge input shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info = GetAxesDimsMappingPair(output_axes_vec, output_specs);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // step2.2: infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> input_dims_mapping =
      GetDimsMappingForAxes(input_axes, axis_to_dim_map);

  // initialize output dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr input_dist_attr =
      CopyTensorDistAttrForOutput(input_specs[0].dist_attr());
  input_dist_attr.set_dims_mapping(input_dims_mapping);

  // Step3  Handle partial (TODO)

  VLOG(4) << "TransposeSPMDRule InferBackward:";
  VLOG(4) << "Output shape: [" << str_join(output_specs[0].shape()) << "] "
          << "dims_mapping: [" << str_join(output_specs[0].dims_mapping())
          << "]";
  VLOG(4) << "Perm: ["
          << str_join(ExtractAttr<std::vector<int64_t>>("perm", attrs)) << "]";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "dims_mapping: [" << str_join(input_dims_mapping) << "]";
  }
  VLOG(4) << std::endl;

  return {{input_dist_attr}, {output_specs[0].dist_attr()}};
}

std::string TransposeSPMDRule::GetOutputNotation(
    int64_t input_ndim,
    const std::string& input_axes,
    const paddle::framework::AttributeMap& attrs) {
  std::vector<int64_t> perm_dims =
      ExtractAttr<std::vector<int64_t>>("perm", attrs);

  // convert the negative dim value to normal dim value
  for (int64_t i = 0, n = perm_dims.size(); i < n; ++i) {
    if (perm_dims[i] < 0) {
      perm_dims[i] = input_ndim + perm_dims[i];
    }
  }

  std::string output_axes = "";
  for (int64_t i = 0; i < input_ndim; i++) {
    output_axes.append(1, input_axes[perm_dims[i]]);
  }

  return output_axes;
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
