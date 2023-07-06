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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/elementwise_spmd_rule.h"
#include "paddle/phi/core/distributed/auto_parallel/utils.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {
using phi::distributed::auto_parallel::str_join;

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ElementwiseSPMDRule::InferForward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  // step0: Verify Input Args Based on Elementwise Logic
  int64_t ninputs = input_specs.size();
  PADDLE_ENFORCE_GT(
      ninputs,
      0,
      phi::errors::InvalidArgument("The size of InputSpec in elementwise must "
                                   "be greater than 0, but got [%d].",
                                   ninputs));
  VerifySpecs(input_specs, "elementwise");

  // step1: Build Einsum Notation
  std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
  std::vector<std::string> input_axes_vec;
  int64_t max_ndim = 0;
  for (int64_t i = 0; i < ninputs; ++i) {
    int64_t ndim = input_specs[i].shape().size();
    if (ndim > max_ndim) {
      max_ndim = ndim;
    }
  }

  // get einsum notation for each input, deal with broadcast
  std::vector<int64_t> broadcast_axis_count(max_ndim, 0);
  for (int64_t i = 0; i < ninputs; ++i) {
    std::vector<int64_t> shape = input_specs[i].shape();
    int64_t ndim = shape.size();
    int64_t start_dim = max_ndim - ndim;
    std::string axes_notation = GetBroadcastAxes(ndim, max_ndim, alphabet);
    if (ninputs > 1) {
      for (int64_t idim = 0; idim < max_ndim; idim++) {
        // deal with the broadcast axes, record the
        // input number at each broadcast axis
        if (idim < start_dim) {
          broadcast_axis_count[idim] += 1;
        } else if (shape[idim - start_dim] == 1) {
          broadcast_axis_count[idim] += 1;
          // mark the broadcast axis to a special "1"
          axes_notation[idim - start_dim] = '1';
        }
      }
    }
    input_axes_vec.emplace_back(axes_notation);
  }

  // get einsum notation for output
  std::string output_axes = GetBroadcastAxes(max_ndim, max_ndim, alphabet);
  for (int64_t idim = 0; idim < max_ndim; idim++) {
    // if all inputs broadcast at this dimension,
    // mark this axis in output as broadcast
    if (broadcast_axis_count[idim] == ninputs) {
      output_axes[idim] = '1';
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

  std::vector<TensorDistAttr> new_input_dist_attrs;
  std::vector<TensorDistAttr> output_dist_attrs;

  // step2.3: update inputs' dims mapping with merged one.
  for (int64_t i = 0; i < ninputs; i++) {
    const DistTensorSpec& spec = input_specs[i];
    TensorDistAttr dist_attr(spec.dist_attr());
    std::vector<int64_t> new_dims_mapping =
        GetDimsMappingForAxes(input_axes_vec[i], axis_to_dim_map);
    dist_attr.set_dims_mapping(new_dims_mapping);
    new_input_dist_attrs.emplace_back(dist_attr);
  }

  // step2.4: handle partial
  // Step2.3.2  handle input tensor partial (TODO)
  VLOG(4) << "ElementwiseSPMDRule InferForward:";
  for (int64_t i = 0; i < ninputs; i++) {
    VLOG(4) << "Input" << std::to_string(i) << " shape: ["
            << str_join(input_specs[i].shape()) << "] "
            << "src_dims_mapping: [" << str_join(input_specs[i].dims_mapping())
            << "] "
            << "dst_dims_mapping: ["
            << str_join(new_input_dist_attrs[i].dims_mapping()) << "]";
  }
  VLOG(4) << "Output dims_mapping: [" + str_join(output_dims_mapping) + "]\n\n";

  output_dist_attrs.emplace_back(output_dist_attr);
  return {new_input_dist_attrs, output_dist_attrs};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ElementwiseSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& output_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of ElementwiseSPMDRule is NOT implemented yet."));

  return {};
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
