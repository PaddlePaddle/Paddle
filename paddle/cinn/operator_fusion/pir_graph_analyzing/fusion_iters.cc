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
  return "LoopIters: " + PrintFusionIters(loop_iters) +
         ", reduce_iter = " + std::to_string(reduce_iter_nums);
}

std::string FusionItersManager::PrintItersSignature(
    const FusionItersSignature& sig) {
  std::stringstream ss;
  ss << "FusionIters Signature:";
  ss << "\n    loop    : " << PrintFusionIters(sig.loop_iters)
     << ", reduce_iter_nums: " << sig.reduce_iter_nums;
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
      symbol::DimExpr dim_expr =
          shape_analysis_->GetProductDimExpr(value, {static_cast<int>(i)});
      if (shape_analysis_->IsEqual(dim_expr, symbol::DimExpr(0))) {
        dim_expr = symbol::DimExpr(1);
      }
      iter2dimexpr_[value_iters[i]] = dim_expr;
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
  if (axes.reduce_size > 0) {
    PADDLE_ENFORCE_LE(
        axes.reduce_size,
        GetCompitableRank(op->operand(0).source()),
        ::common::errors::InvalidArgument("The number of reduce_axis should be "
                                          "no more than output value ranks."));
  }
  FusionItersSignature result;
  result.loop_iters = axes.loop.axis_names;
  result.reduce_iter_nums = axes.reduce_size;
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

FusionItersSignature FusionItersManager::SingleDownstreamItersFusion(
    const FusionItersSignature& upstream,
    const FusionItersSignature& downstream) {
  VLOG(4) << "[ItersFusion] Start SingleDownstreamItersFusion."
          << "\nUpstream: " << PrintItersSignature(upstream)
          << "\nDownstream: " << PrintItersSignature(downstream);
  FusionItersSignature fused_signature;

  const auto [upstream_non_reduce_iters, upstream_reduce_iters] =
      SplitReduceIters(upstream);
  const auto [downstream_non_reduce_iters, downstream_reduce_iters] =
      SplitReduceIters(downstream);

  if (upstream_reduce_iters.empty()) {
    // Trivial Sink
    fused_signature.loop_iters = downstream.loop_iters;
    fused_signature.reduce_iter_nums = downstream.reduce_iter_nums;
  } else if (downstream_reduce_iters.empty()) {
    // Reduce x Trivial Fusion
    const auto [shared_iters, _UNUSED] = SplitFirstWhetherInSecond(
        downstream_non_reduce_iters, upstream_non_reduce_iters);
    fused_signature.loop_iters =
        ConcatVector(shared_iters, upstream_reduce_iters);
    fused_signature.reduce_iter_nums = upstream.reduce_iter_nums;
  } else {
    // Reduce x Reduce Fusion
    PADDLE_ENFORCE_EQ(
        upstream.reduce_iter_nums,
        downstream.reduce_iter_nums,
        ::common::errors::InvalidArgument(
            "The number of reduce axis in RR Fusion should be equal."));
    fused_signature.loop_iters = downstream.loop_iters;
    fused_signature.reduce_iter_nums = downstream.reduce_iter_nums;
  }

  PADDLE_ENFORCE_EQ(
      upstream.output_values.size(),
      1,
      ::common::errors::InvalidArgument(
          "Node in single downstream fusion should have only one output."));
  // TODO(huangjiyi): fix upstream output have multi usage in one downstream
  // PADDLE_ENFORCE_EQ(--value_remain_usage_[*upstream.output_values.begin()],
  //                   0,
  //                   ::common::errors::InvalidArgument(
  //                       "Upstream should have one downstream."));
  fused_signature.input_values =
      SetUnion(upstream.input_values,
               SetDifference(downstream.input_values, upstream.output_values));
  fused_signature.output_values = downstream.output_values;

  VLOG(4) << "[ItersFusion] Fused: \n" << PrintItersSignature(fused_signature);
  return fused_signature;
}

FusionItersSignature FusionItersManager::MultiDownstreamItersFusion(
    const FusionItersSignature& upstream,
    const FusionItersSignature& downstream,
    const FusionItersManager::FusionDirection& direction) {
  VLOG(4) << "[ItersFusion] Start MultiDownstreamItersFusion."
          << "\nUpstream: " << PrintItersSignature(upstream)
          << "\nDownstream: " << PrintItersSignature(downstream);
  FusionItersSignature fused_signature;

  const auto [upstream_non_reduce_iters, upstream_reduce_iters] =
      SplitReduceIters(upstream);
  const auto [downstream_non_reduce_iters, downstream_reduce_iters] =
      SplitReduceIters(downstream);
  fused_signature.loop_iters = direction == FusionDirection::upstream2downstream
                                   ? downstream.loop_iters
                                   : upstream.loop_iters;
  if (upstream_reduce_iters.empty() && downstream_reduce_iters.empty()) {
    // Trivial x Trivial Fusion
    fused_signature.reduce_iter_nums = 0;
  } else if (upstream_reduce_iters.empty()) {
    // Trivial x Reduce Fusion
    fused_signature.reduce_iter_nums = downstream.reduce_iter_nums;
  } else {
    // Reduce x Reduce Fusion + Reduce x Others Fusion
    fused_signature.reduce_iter_nums = upstream.reduce_iter_nums;
  }

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

bool FusionItersManager::IterSymbolEqual(const std::string& lhs,
                                         const std::string& rhs) {
  PADDLE_ENFORCE(iter2dimexpr_.count(lhs) && iter2dimexpr_.count(rhs),
                 ::common::errors::InvalidArgument(
                     "Cannot found symbol of input iter %s or %s", lhs, rhs));
  return shape_analysis_->IsEqual(iter2dimexpr_[lhs], iter2dimexpr_[rhs]);
}

bool FusionItersManager::IterSymbolEqualOne(const std::string& sym) {
  return shape_analysis_->IsEqual(iter2dimexpr_[sym], 1);
}

std::pair<FusionIters, FusionIters> SplitReduceIters(
    const FusionItersSignature& sig) {
  const size_t rank = sig.loop_iters.size();
  return {SliceVector(sig.loop_iters, 0, rank - sig.reduce_iter_nums),
          SliceVector(sig.loop_iters, rank - sig.reduce_iter_nums, rank)};
}

}  // namespace cinn::fusion
