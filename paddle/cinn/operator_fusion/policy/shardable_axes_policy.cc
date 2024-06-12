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

#include "paddle/cinn/operator_fusion/policy/shardable_axes_policy.h"
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

template <typename T>
bool ShardableAxesRRFusePolicy<T>::IsDownstreamStmtDependReduceOp(
    pir::Operation* reduce, const StmtPattern<T>& downstream) {
  const auto& values = GetPatternInputValues(downstream);
  for (const auto& value : reduce->results()) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::optional<ReducePattern<T>>
ShardableAxesRRFusePolicy<T>::GetDownstreamFromCandidate(
    const ReducePattern<T>& upstream,
    const std::vector<ReducePattern<T>>& candidates) {
  pir::Operation* reduce = upstream.GetReduceOp();
  for (const auto& candidate : candidates) {
    if (IsDownstreamStmtDependReduceOp(reduce, candidate)) {
      return candidate;
    }
  }
  return {};
}

static std::set<std::string> GetReduceAxesName(
    const ShardableAxesSignature& signature) {
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> res(input_names.begin(), input_names.end());
  for (const auto& n : output_names) {
    res.erase(n);
  }
  return res;
}

template <typename T>
bool ShardableAxesRRFusePolicy<T>::ReduceTreeGrownCanMerge(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  if (!upstream->IsReduceTree() || !downstream->IsReduceTree()) {
    return false;
  }
  const auto& upstream_tree =
      std::get<ReduceTreePattern>(upstream->stmt_pattern());
  const auto& downstream_tree =
      std::get<ReduceTreePattern>(downstream->stmt_pattern());
  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.reduce_patterns_);
  if (!maybe_downstream_op.has_value()) {
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();
  const auto& reduce_names =
      GetReduceAxesName(axes_info_.GetSignature(downstream_reduce_op));
  for (const auto& n :
       axes_info_.GetAxes(downstream_reduce_op->result(0)).axis_names) {
    if (reduce_names.count(n) > 0) {
      // not meeting the BroadcastEdge condition.
      return false;
    }
  }
  return true;
}

template <typename T>
bool ShardableAxesRRFusePolicy<T>::CanFuse(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  // TODO(wuzhanfei) shardable axes policy
  return ReduceTreeGrownCanMerge(upstream, downstream);
}

template class RelativeJudgePolicy<FrontendStage>;
template class RelativeJudgePolicy<BackendStage>;

}  // namespace cinn::fusion
