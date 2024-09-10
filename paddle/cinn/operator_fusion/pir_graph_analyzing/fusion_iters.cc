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

std::string PrintFusionIters(const FusionIters& iters) {
  return "[ " + cinn::utils::Join(iters, ",") + " ]";
}

std::string FusionItersSignature::DebugStr() const {
  return "LoopIters: " + PrintFusionIters(loop_iters);
}

std::string FusionItersManager::PrintItersSignature(
    const FusionItersSignature& sig) {
  std::stringstream ss;
  ss << "FusionIters Signature:";
  ss << "\n    loop    : " << PrintFusionIters(sig.loop_iters);
  size_t count = 0;
  for (const auto& value : sig.input_values) {
    ss << "\n    input  " << count++ << ": "
       << PrintFusionIters(value2iters_[value]) << " (" << value.impl() << ")";
  }
  count = 0;
  for (const auto& value : sig.output_values) {
    ss << "\n    output " << count++ << ": "
       << PrintFusionIters(value2iters_[value]) << " (" << value.impl() << "), "
       << "remain_usages: " << value_remain_usage_[value];
  }
  return ss.str();
}

void FusionItersManager::StoreIter2DimExprForValue(const pir::Value& value) {
  PADDLE_ENFORCE_NE(value2iters_.count(value),
                    0,
                    ::common::errors::InvalidArgument(
                        "Can not find target value in value2iters_ map."));
  const auto& value_iters = value2iters_[value];
  for (size_t i = 0; i < value_iters.size(); ++i) {
    if (iter2dimexpr_.count(value_iters[i]) == 0) {
      iter2dimexpr_[value_iters[i]] =
          shape_analysis_->GetProductDimExpr(value, {static_cast<int>(i)});
    }
  }
}

FusionItersSignature FusionItersManager::GetItersSignature(pir::Operation* op) {
  const auto& axes = axes_info_->GetModifiedSignature(op);
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
  FusionItersSignature result;
  result.loop_iters = axes.loop.axis_names;
  result.input_values = ToSet(op->operands_source());
  result.output_values = ToSet(op->results());

  for (size_t i = 0; i < op->num_operands(); ++i) {
    const auto& value = op->operand_source(i);
    if (value2iters_.count(value) == 0) {
      value2iters_[value] = axes.inputs[i].axis_names;
      value_remain_usage_[value] = value.use_count();
      StoreIter2DimExprForValue(value);
    }
  }
  for (size_t i = 0; i < op->num_results(); ++i) {
    const auto& value = op->result(i);
    if (value2iters_.count(value) == 0) {
      value2iters_[value] = axes.outputs[i].axis_names;
      value_remain_usage_[value] = value.use_count();
      StoreIter2DimExprForValue(value);
    }
  }
  return result;
}

FusionItersSignature FusionItersManager::FuseItersSignature(
    const FusionItersSignature& upstream,
    const FusionItersSignature& downstream,
    bool is_sink) {
  VLOG(4) << "[ItersFusion] Start FuseItersSignature."
          << "\nis_sink: " << (is_sink ? "True" : "False")
          << "\nUpstream: " << PrintItersSignature(upstream)
          << "\nDownstream: " << PrintItersSignature(downstream);
  FusionItersSignature fused_signature;
  fused_signature.loop_iters =
      is_sink ? downstream.loop_iters : upstream.loop_iters;

  auto link_values =
      SetIntersection(upstream.output_values, downstream.input_values);
  fused_signature.input_values =
      SetUnion(upstream.input_values,
               SetDifference(downstream.input_values, link_values));
  fused_signature.output_values =
      SetUnion(downstream.output_values,
               SetDifference(upstream.output_values, link_values));
  for (const auto& link_value : link_values) {
    if (--value_remain_usage_[link_value] > 0) {
      fused_signature.output_values.insert(link_value);
    }
  }

  VLOG(4) << "[ItersFusion] Fused: \n" << PrintItersSignature(fused_signature);
  return fused_signature;
}

}  // namespace cinn::fusion
