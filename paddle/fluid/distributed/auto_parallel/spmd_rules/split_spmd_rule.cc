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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/split_spmd_rule.h"
#include <algorithm>
#include <typeinfo>
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SplitSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                            const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Elementwise Logic
  int64_t ninputs = input_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in split must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  VerifySpecs(input_specs, "split");

  // step1: Build Einsum Notation
  int64_t ndim = input_specs[0].shape().size();
  int64_t noutput = 0;
  // split api uses num or sections as attribute
  if (attrs.find("num") != attrs.end()) {
    noutput = ExtractAttr<int64_t>("num", attrs);
  } else if (attrs.find("sections") != attrs.end()) {
    std::vector<int64_t> sections =
        ExtractAttr<std::vector<int64_t>>("sections", attrs);
    noutput = sections.size();
  }
  int64_t axis = ExtractAttr<int>("axis", attrs);
  if (axis < 0) {
    axis += ndim;
  }
  std::string alphabet = "abcdefghijlmnopqrstuvwxyz";

  // get einsum notation for input, use a special
  // notation 'k' to mark the splitted axis in input
  std::vector<std::string> input_axes_vec;
  std::string input_axes = alphabet.substr(0, ndim);
  input_axes[axis] = 'k';
  input_axes_vec.emplace_back(input_axes);

  // get einsum notation for output
  std::string output_axes(input_axes);
  // the splitted axis cannot be sharded, set its notation
  // with the special '1' to set its dim mapping to -1.
  output_axes[axis] = '1';

  // step2: Sharding Propogation
  // step2.1: merge input shardings
  std::vector<std::pair<std::string, std::vector<int64_t>>> axes_sharding_info;
  axes_sharding_info = GetAxesDimsMappingPair(input_axes_vec, input_specs);
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors(axes_sharding_info);

  // step2.2: infer output dimsmapping from merged input dimsmapping
  std::vector<int64_t> output_dims_mapping =
      GetDimsMappingForAxes(output_axes, axis_to_dim_map);

  // get the dist attributes for all outputs, the
  // dist attributes are same for all outputs.
  std::vector<TensorDistAttr> output_dist_attrs;
  for (int64_t i = 0; i < noutput; i++) {
    output_dist_attrs.emplace_back(
        CopyTensorDistAttrForOutput(input_specs[0].dist_attr()));
    output_dist_attrs[i].set_dims_mapping(output_dims_mapping);
  }

  // step2.3 get new dist attribute for input. the splitted
  // cannot be sharded, if it is sharded, set it to replicated.
  std::vector<TensorDistAttr> new_input_dist_attrs;
  new_input_dist_attrs.emplace_back(input_specs[0].dist_attr());
  std::vector<int64_t> new_input_dims_mapping(input_specs[0].dims_mapping());
  new_input_dims_mapping[axis] = -1;
  new_input_dist_attrs[0].set_dims_mapping(new_input_dims_mapping);

  // Step2.4  handle input tensor partial (TODO)
  VLOG(4) << "SplitSPMDRule InferForward: ";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "einsum_notation: " << input_axes << " src_dims_mapping: ["
            << str_join(input_specs[i].dims_mapping()) << "] "
            << "dst_dims_mapping: ["
            << str_join(new_input_dist_attrs[i].dims_mapping()) << "]";
  }
  for (int64_t i = 0; i < noutput; i++) {
    VLOG(4) << "Output" << std::to_string(i) << " dims_mapping: ["
            << str_join(output_dims_mapping) << "]";
  }

  return {new_input_dist_attrs, output_dist_attrs};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
SplitSPMDRule::InferBackward(const std::vector<DistTensorSpec>& input_specs,
                             const std::vector<DistTensorSpec>& output_specs,
                             const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of SplitPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
