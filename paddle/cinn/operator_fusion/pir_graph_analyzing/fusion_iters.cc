// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/cinn/operator_fusion/pir_graph_analyzing/fusion_iters.h"
#include "paddle/cinn/operator_fusion/pattern_node.h"

namespace cinn::fusion {

FusionItersSignature::FusionItersSignature(pir::Operation* op,
                                           const ShardableAxesSignature& axes) {
  PADDLE_ENFORCE_EQ(
      axes.inputs.size(),
      op->num_operands(),
      ::common::errors::InvalidArgument("The number of input_iters should be "
                                        "equal to the number of operands."));
  PADDLE_ENFORCE_EQ(
      axes.outputs.size(),
      op->num_results(),
      ::common::errors::InvalidArgument("The number of output_iters should be "
                                        "equal to the number of results."));
  loop_iters = axes.loop.axis_names;
  for (const auto& iters : axes.inputs) {
    input_iters.push_back(iters.axis_names);
  }
  for (const auto& iters : axes.outputs) {
    output_iters.push_back(iters.axis_names);
  }
  input_values = op->operands_source();
  output_values = op->results();
}

std::string PrintFusionIters(const FusionIters& iters) {
  std::stringstream ss;
  ss << "[ ";
  for (const auto& iter : iters) {
    ss << iter << ",";
  }
  return ss.str().substr(0, ss.str().size() - 1) + " ]";
}

std::string FusionItersSignature::DebugStr() const {
  std::stringstream ss;
  ss << "FusionIters Signature:";
  ss << "\n    loop    : " << PrintFusionIters(loop_iters);
  for (size_t i = 0; i < input_iters.size(); ++i) {
    ss << "\n    input  " << i << ": " << PrintFusionIters(input_iters[i]);
  }
  for (size_t i = 0; i < output_iters.size(); ++i) {
    ss << "\n    output " << i << ": " << PrintFusionIters(output_iters[i]);
  }
  return ss.str();
}

FusionItersSignature SingleDownstreamItersFusion(PatternNodePtr upstream,
                                                 PatternNodePtr downstream,
                                                 bool is_sink) {
  VLOG(4) << "[ItersFusion] Start SingleDownstreamItersFusion.";
  auto upstream_iters = upstream->fusion_iters();
  auto downstream_iters = downstream->fusion_iters();
  PADDLE_ENFORCE_EQ(upstream_iters.output_iters.size(),
                    1,
                    ::common::errors::InvalidArgument(
                        "The number of upstream outputs should be 1."));

  FusionItersSignature fused_iters;
  fused_iters.loop_iters =
      is_sink ? downstream_iters.loop_iters : upstream_iters.loop_iters;
  fused_iters.output_values = downstream_iters.output_values;
  fused_iters.output_iters = downstream_iters.output_iters;

  const auto& upstream_output_value = upstream_iters.output_values[0];
  for (size_t i = 0; i < downstream_iters.input_values.size(); ++i) {
    if (downstream_iters.input_values[i] == upstream_output_value) {
      for (size_t j = 0; j < upstream_iters.input_iters.size(); ++j) {
        fused_iters.input_iters.push_back(upstream_iters.input_iters[j]);
        fused_iters.input_values.push_back(upstream_iters.input_values[j]);
      }
    } else {
      fused_iters.input_iters.push_back(downstream_iters.input_iters[i]);
      fused_iters.input_values.push_back(downstream_iters.input_values[i]);
    }
  }
  VLOG(4) << "[ItersFusion] End SingleDownstreamItersFusion.";
  return fused_iters;
}

}  // namespace cinn::fusion
