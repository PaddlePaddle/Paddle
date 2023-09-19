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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/reduction_spmd_rule.h"
#include <algorithm>
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

using phi::distributed::auto_parallel::str_join;

std::string ReductionSPMDRule::GetOutputNotation(
    int64_t input_ndim,
    const std::string& input_axes,
    const paddle::framework::AttributeMap& attrs) {
  bool keep_dim = ExtractAttr<bool>("keep_dim", attrs);
  std::vector<int64_t> reduce_dims =
      ExtractAttr<std::vector<int64_t>>("axis", attrs);

  // convert the negative dim value to normal dim value
  for (auto& reduce_dim : reduce_dims) {
    if (reduce_dim < 0) {
      reduce_dim = input_ndim + reduce_dim;
    }
  }

  std::string output_axes = "";
  for (int64_t i = 0; i < input_ndim; i++) {
    std::vector<int64_t>::iterator iter =
        std::find(reduce_dims.begin(), reduce_dims.end(), i);
    if (iter != reduce_dims.end()) {
      // if i is reduce dim, the corresponding input axis
      // will not be appended at the end of output_axes
      if (keep_dim) {
        output_axes.append(1, '1');
      }
    } else {
      // otherwise, the corresponding input axis
      // will be appended at the end of output_axes
      output_axes.append(1, input_axes[i]);
    }
  }

  return output_axes;
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ReductionSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                                const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Elementwise Logic
  int64_t ninputs = input_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in reduction must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  VerifySpecs(input_specs, "reduction");

  // step1: Build Einsum Notation
  // get einsum notation for input
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  int64_t ndim = input_specs[0].shape().size();
  std::vector<std::string> input_axes_vec;
  std::string input_axes = alphabet.substr(0, ndim);
  input_axes_vec.emplace_back(input_axes);

  // get einsum notation for output
  std::string output_axes = GetOutputNotation(ndim, alphabet, attrs);

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

  // step3: handle partial
  // Step3.1 Output Partial
  std::vector<int64_t> partial_on_dims =
      ResoluteOutputPartialDimension(axis_to_dim_map, output_axes);
  output_dist_attr.set_partial_status(
      partial_on_dims /*, handle reduce_type in future  */);

  std::vector<TensorDistAttr> output_dist_attrs;
  output_dist_attrs.emplace_back(output_dist_attr);

  // Step3.2  handle input tensor partial (TODO)
  // If the op is a linear op, i.e. `linearity` is true, it supports
  // the input to be partial. Otherwise, the input cannot be partial
  // on reduced axes, we should reshard the input when the reduced
  // axes are parital.
  VLOG(4) << "ReductionSPMDRule InferForward: ";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "src_dims_mapping: [" << str_join(input_specs[i].dims_mapping())
            << "] "
            << "dst_dims_mapping: [" << str_join(input_specs[i].dims_mapping())
            << "]";
  }
  VLOG(4) << "Output dims_mapping: [" + str_join(output_dims_mapping) + "] "
          << "partial_on_dims: [" + str_join(partial_on_dims) + "]\n\n";

  return {{input_specs[0].dist_attr()}, output_dist_attrs};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ReductionSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& input_specs,
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Elementwise Logic
  int64_t ninputs = input_specs.size();
  int64_t noutputs = output_specs.size();
  PADDLE_ENFORCE_EQ(
      ninputs,
      1,
      phi::errors::InvalidArgument("The size of InputSpec in reduction must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  PADDLE_ENFORCE_EQ(
      noutputs,
      1,
      phi::errors::InvalidArgument("The size of OutputSpec in reduction must "
                                   "be equal to 1, but got [%d].",
                                   ninputs));
  VerifySpecs(output_specs, "reduction_backward");

  // step1: Build Einsum Notation
  // get einsum notation for input
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  int64_t ndim = input_specs[0].shape().size();
  std::string input_axes = alphabet.substr(0, ndim);

  // get einsum notation for output
  std::string output_axes = GetOutputNotation(ndim, alphabet, attrs);

  // step2: Sharding Propogation
  std::unordered_map<std::string, int64_t> axis_to_dim_map =
      ShardingMergeForTensors({{output_axes, output_specs[0].dims_mapping()}});

  // step2.2: infer input dims mapping from output dims mapping
  std::vector<int64_t> input_dims_mapping =
      GetDimsMappingForAxes(input_axes, axis_to_dim_map, true);

  // initialize input dist_attr's process_mesh, batch_dim and dynamic dims with
  // input dist_attr.
  TensorDistAttr input_dist_attr(input_specs[0].dist_attr());
  input_dist_attr.set_dims_mapping(input_dims_mapping);

  // step3: handle partial (TODO)

  VLOG(4) << "ReductionSPMDRule InferBackward: ";
  VLOG(4) << "Output shape:[" << str_join(output_specs[0].shape())
          << "] dims_mapping: [" << str_join(output_specs[0].dims_mapping())
          << "]";
  VLOG(4) << "Input0: "
          << " shape: [" << str_join(input_specs[0].shape()) << "] "
          << "dims_mapping: [" << str_join(input_dist_attr.dims_mapping())
          << "]";

  return {{input_dist_attr}, {output_specs[0].dist_attr()}};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
