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

#include "paddle/fluid/distributed/auto_parallel/spmd_rules/replicated_spmd_rule.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ReplicatedSPMDRule::InferForward(const std::vector<DistTensorSpec>& input_specs,
                                 const paddle::framework::AttributeMap& attrs) {
  std::vector<TensorDistAttr> intput_dist_attrs;
  std::vector<TensorDistAttr> output_dist_attrs;
  intput_dist_attrs.reserve(input_specs.size());

  for (auto& input_spec : input_specs) {
    intput_dist_attrs.push_back(ReplicatedOnMesh(input_spec.dist_attr()));
  }

  // TODO(ljz): we need to know num of output and size of each output before
  // generate the excat replicasted dist tensor attr for the current op.
  // here we just assume that only one output tensor and has the same size as
  // the first input tensor.
  return {intput_dist_attrs, {ReplicatedOnMesh(input_specs[0].dist_attr())}};
}

std::pair<std::vector<TensorDistAttr>, std::vector<TensorDistAttr>>
ReplicatedSPMDRule::InferBackward(
    const std::vector<DistTensorSpec>& input_specs,
    const paddle::framework::AttributeMap& attrs) {
  PADDLE_THROW(phi::errors::Unimplemented(
      "InferBackward of ReplicatedSPMDRule is NOT implemented yet."));
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
