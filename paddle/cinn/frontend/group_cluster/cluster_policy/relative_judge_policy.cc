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

#include "paddle/cinn/frontend/group_cluster/cluster_policy/relative_judge_policy.h"

namespace cinn::frontend::group_cluster::policy {
size_t ValueDim::GetNumbericValue() const {
  return v_.type().dyn_cast<pir::DenseTensorType>().dims().at(idx_);
}

bool RelativeJudgePolicy::IsDownstreamStmtDependReduceOp(
    pir::Operation* reduce, const StmtPattern& downstream) {
  const auto& values = GetPatternInputValues(downstream);
  for (const auto& value : reduce->results()) {
    if (std::find(values.begin(), values.end(), value) != values.end()) {
      return true;
    }
  }
  return false;
}

std::optional<ReducePattern> RelativeJudgePolicy::GetDownstreamFromCandidate(
    const ReducePattern& upstream,
    const std::vector<ReducePattern>& candidates) {
  pir::Operation* reduce = upstream.GetReduceOp();
  for (const auto& candidate : candidates) {
    if (IsDownstreamStmtDependReduceOp(reduce, candidate)) {
      return candidate;
    }
  }
  return {};
}

SplitedDims SplitReduceDims(const ShardableAxesSignature& signature,
                            const pir::Value& v) {
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());
  auto result = SplitedDims();
  int idx = 0;
  for (const auto& in : input_names) {
    if (output_names_set.count(in) == 0) {
      result.non_related.emplace_back(v, idx);
    } else {
      result.related.emplace_back(v, idx);
    }
    idx += 1;
  }
  return result;
}

bool RelativeJudgePolicy::IsBroadcastEdge(
    const std::vector<ValueDim>& upstream_out_dims,
    const std::vector<ValueDim>& downstream_reduce_dims) {
  for (const auto& downstream_reduce_dim : downstream_reduce_dims) {
    for (const auto& upstream_out_dim : upstream_out_dims) {
      if (IsRelated(upstream_out_dim, downstream_reduce_dim)) {
        return false;
      }
    }
  }
  return true;
}

bool RelativeJudgePolicy::ReduceTreeGrownCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  if (!upstream->IsReduceTree() || !downstream->IsReduceTree()) {
    return false;
  }
  const auto& upstream_tree =
      std::get<ReduceTreePattern>(upstream->stmt_pattern_);
  const auto& downstream_tree =
      std::get<ReduceTreePattern>(downstream->stmt_pattern_);
  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.reduce_patterns_);
  if (!maybe_downstream_op.has_value()) {
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();
  const auto& split_reduce_dim_result =
      SplitReduceDims(axes_info_.GetSignature(downstream_reduce_op),
                      downstream_reduce_op->result(0));
  const auto& upstream_output_dims = GetAllValueDimFromValue(reduce_out_value);
  return IsBroadcastEdge(upstream_output_dims,
                         split_reduce_dim_result.non_related);
}

SplitedDims RelativeJudgePolicy::SplitDimsWithRelationship(
    const std::vector<ValueDim>& targets,
    const std::vector<ValueDim>& related_with) {
  auto result = SplitedDims();
  bool is_related;

  for (auto& target_dim : targets) {
    is_related = false;
    for (auto& related_dim : related_with) {
      if (IsRelated(target_dim, related_dim)) is_related = true;
    }
    if (is_related) {
      result.related.push_back(target_dim);
    } else {
      result.non_related.push_back(target_dim);
    }
  }

  return result;
}

bool DimsEquel(const std::vector<ValueDim>& first,
               const std::vector<ValueDim>& second) {
  const auto GetDimInfo =
      [](const std::vector<ValueDim>& dims) -> std::unordered_map<size_t, int> {
    std::unordered_map<size_t, int> result;
    for (const auto& dim : dims) {
      size_t value = dim.GetNumbericValue();
      if (result.find(value) == result.end()) {
        result[value] = 1;
      } else {
        result[value] += 1;
      }
    }
    return result;
  };

  const std::unordered_map<size_t, int>& first_dims = GetDimInfo(first);
  const std::unordered_map<size_t, int>& second_dims = GetDimInfo(second);
  if (first_dims.size() != second_dims.size()) return false;
  for (const auto& [dim_value, count] : first_dims) {
    if (second_dims.find(dim_value) == second_dims.end() ||
        second_dims.at(dim_value) != count)
      return false;
  }
  return true;
}

bool RelativeJudgePolicy::ReducePlusTrivialCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  if (!upstream->IsReduceTree() || !downstream->IsTrivial()) {
    return false;
  }

  const auto& split_reduce_dims_result =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op_),
                      upstream->sink_op_->result(0));

  const auto& upstream_reduce_dims = split_reduce_dims_result.non_related;
  const auto& upstream_non_reduce_dims = split_reduce_dims_result.related;

  const auto& all_trivial_output_dims =
      GetAllValueDimFromValue(downstream->sink_op_->result(0));

  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      all_trivial_output_dims, upstream_non_reduce_dims);

  return DimsEquel(split_trivial_dims_result.non_related, upstream_reduce_dims);
}

bool RelativeJudgePolicy::CanFuse(const PatternNodePtr& upstream,
                                  const PatternNodePtr& downstream) {
  return ReduceTreeGrownCanMerge(upstream, downstream) ||
         ReducePlusTrivialCanMerge(upstream, downstream);
}

PatternNodePtr Merge(const PatternNodePtr& upstream,
                     const PatternNodePtr& downstream) {
  return nullptr;
}

}  // namespace cinn::frontend::group_cluster::policy
