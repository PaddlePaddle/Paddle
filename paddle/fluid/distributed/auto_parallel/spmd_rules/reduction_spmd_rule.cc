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
  bool keep_dim = ExtractAttr<bool>("keep_dim", attrs);
  std::vector<int64_t> reduce_dims =
      ExtractAttr<std::vector<int64_t>>("axis", attrs);
  bool linearity = ExtractAttr<bool>("linearity", attrs);
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

  // get einsum notation for input
  int64_t ndim = input_specs[0].shape().size();
  std::vector<std::string> input_axes_vec;
  std::string input_axes = alphabet.substr(0, ndim);
  input_axes_vec.emplace_back(input_axes);

  // get einsum notation for output
  for (int64_t i = 0, n = reduce_dims.size(); i < n; ++i) {
    // convert the negative dim value to normal dim value
    if (reduce_dims[i] < 0) {
      reduce_dims[i] = ndim + reduce_dims[i];
    }
  }
  std::string output_axes = "";
  for (int64_t i = 0; i < ndim; i++) {
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

  std::vector<TensorDistAttr> output_dist_attrs;
  output_dist_attrs.emplace_back(output_dist_attr);

  // step2.4: handle partial
  // Step2.4.1 Output Partial
  // If the op is a linear op, i.e. `linearity` is true, the output's
  // dims mapping can be partial. Otherwise, partial is not supported,
  // the input should be reshard when its sharded axes are reduced,
  // reduced, i.e., we should replicate the crossponding axes when
  // when the op is not linear.
  std::vector<TensorDistAttr> new_input_dist_attrs;
  std::vector<int64_t> new_input_dims_mapping(input_specs[0].dims_mapping());
  std::vector<int64_t> partial_on_dims;
  if (linearity) {
    // get the partial states for output
    partial_on_dims =
        ResoluteOutputPartialDimension(axis_to_dim_map, output_axes);
  } else {
    // for non-linear ops, we need to reshard the
    // input when the sharded axes are reduced
    for (int64_t i = 0; i < ndim; i++) {
      std::vector<int64_t>::iterator iter =
          std::find(reduce_dims.begin(), reduce_dims.end(), i);
      if (iter != reduce_dims.end()) {
        new_input_dims_mapping[i] = -1;
      }
    }
  }

  TensorDistAttr new_input_dist_attr(input_specs[0].dist_attr());
  new_input_dist_attr.set_dims_mapping(new_input_dims_mapping);
  new_input_dist_attrs.emplace_back(new_input_dist_attr);

  // Step2.4.2  handle input tensor partial (TODO)
  VLOG(4) << "ReductionSPMDRule InferForward: "
          << "X shape: [" << str_join(input_specs[0].shape())
          << "], src_dims_mapping: [" << str_join(input_specs[0].dims_mapping())
          << "], dst_dims_mapping: ["
          << str_join(new_input_dist_attrs[0].dims_mapping())
          << "]; Output dims_mapping: [" << str_join(output_dims_mapping)
          << "], partial_on_dims: [" << str_join(partial_on_dims) << "]";

  return {new_input_dist_attrs, output_dist_attrs};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ReductionSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of ReductionSPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
