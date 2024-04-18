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

#include "paddle/cinn/operator_fusion/policy/relative_judge_policy.h"
#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/frontend/pattern.h"

namespace cinn::fusion {

template <typename T>
bool RelativeJudgePolicy<T>::IsDownstreamStmtDependReduceOp(
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
RelativeJudgePolicy<T>::GetDownstreamFromCandidate(
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

std::pair<std::vector<ValueDim>, std::vector<ValueDim>> SplitReduceDims(
    const ShardableAxesSignature& signature, pir::Operation* op) {
  const auto& v = op->operand_source(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());

  std::vector<ValueDim> reduce_dims;
  std::vector<ValueDim> non_reduce_dims;
  auto usage_idx = GetUsageIdx(v, op);

  int idx = 0;
  for (const auto& in : input_names) {
    if (output_names_set.count(in) == 0) {
      reduce_dims.emplace_back(v, idx, usage_idx);
    } else {
      non_reduce_dims.emplace_back(v, idx, usage_idx);
    }
    idx += 1;
  }

  if (VLOG_IS_ON(4)) {
    std::stringstream ss;
    ss << "SplitDims:\nreduce_dims:\n";
    for (const auto& dim : reduce_dims) {
      ss << dim.DebugStr() << "\n";
    }
    ss << "non_reduce_dims:\n";
    for (const auto& dim : non_reduce_dims) {
      ss << dim.DebugStr() << "\n";
    }
    VLOG(4) << ss.str();
  }

  return {reduce_dims, non_reduce_dims};
}

template <typename T>
std::pair<std::vector<ValueDim>, std::vector<ValueDim>>
RelativeJudgePolicy<T>::SplitDimsWithRelationship(
    const std::vector<ValueDim>& targets,
    const std::vector<ValueDim>& related_with) {
  std::vector<ValueDim> related_dims;
  std::vector<ValueDim> non_related_dims;

  bool is_related = false;
  for (auto& target_dim : targets) {
    is_related = false;
    for (auto& related_dim : related_with) {
      if (IsRelated(related_dim, target_dim)) is_related = true;
    }
    if (is_related) {
      related_dims.push_back(target_dim);
    } else {
      non_related_dims.push_back(target_dim);
    }
  }

  if (VLOG_IS_ON(4)) {
    std::stringstream ss;
    ss << "SplitDims:\nrelated_dims:\n";
    for (const auto& dim : related_dims) {
      ss << dim.DebugStr() << "\n";
    }
    ss << "non_related_dims:\n";
    for (const auto& dim : non_related_dims) {
      ss << dim.DebugStr() << "\n";
    }
    VLOG(4) << ss.str();
  }

  return {related_dims, non_related_dims};
}

bool DimsEqual(const std::vector<ValueDim>& first,
               const std::vector<ValueDim>& second) {
  const auto GetDimInfo =
      [](const std::vector<ValueDim>& dims) -> std::unordered_map<size_t, int> {
    std::unordered_map<size_t, int> result;
    for (const auto& dim : dims) {
      VLOG(4) << "dim: " << dim.DebugStr();
      size_t value = dim.GetNumericValue();
      VLOG(4) << "value: " << value;
      if (result.find(value) == result.end()) {
        result[value] = 1;
      } else {
        result[value] += 1;
      }
    }
    return result;
  };
  VLOG(4) << "GetDimInfo";
  const std::unordered_map<size_t, int>& first_dims = GetDimInfo(first);
  VLOG(4) << "GetDimInfo";
  const std::unordered_map<size_t, int>& second_dims = GetDimInfo(second);
  if (first_dims.size() != second_dims.size()) return false;
  for (const auto& [dim_value, count] : first_dims) {
    if (second_dims.find(dim_value) == second_dims.end() ||
        second_dims.at(dim_value) != count)
      return false;
  }
  return true;
}

pir::Operation* FindUserOp(const std::vector<pir::Operation*>& candidates,
                           const pir::Value& value) {
  std::vector<pir::Operation*> results;
  for (auto consumer_it = value.use_begin(); consumer_it != value.use_end();
       ++consumer_it) {
    pir::Operation* user_op = consumer_it.owner();
    auto iter = std::find(candidates.begin(), candidates.end(), user_op);
    if (iter != candidates.end()) {
      results.emplace_back(*iter);
    }
  }
  CHECK(results.size() == 1) << "Zero Or Multi User Op Found In Candidates!";
  return results.front();
}

template <typename T>
bool RelativeJudgePolicy<T>::ReduceTreeGrownCanMerge(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& upstream_tree =
      std::get<ReduceTreePattern<T>>(upstream->stmt_pattern_);
  const auto& downstream_tree =
      std::get<ReduceTreePattern<T>>(downstream->stmt_pattern_);

  VLOG(4) << "upstream->stmt_pattern_:"
          << OpsDebugStr(GetOpsInPattern<T>(upstream_tree));
  VLOG(4) << "downstream->stmt_pattern_"
          << OpsDebugStr(GetOpsInPattern<T>(downstream_tree));

  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.FlattenReducePattern());
  int idx = 0;
  for (const auto& r_pattern : downstream_tree.childs()) {
    idx += 1;
    VLOG(4) << "downstream_tree.reduce_patterns_"
            << "[" << idx << "]" << OpsDebugStr(GetOpsInPattern<T>(r_pattern));
  }
  if (!maybe_downstream_op.has_value()) {
    VLOG(4) << "can't find candidate from patterns. can fuse return false.";
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  auto downstream_connect_op =
      FindUserOp(downstream_tree.ops(), reduce_out_value);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();

  const auto& [downstream_reduce_dims, _UNUSED] = SplitReduceDims(
      axes_info_.GetSignature(downstream_reduce_op), downstream_reduce_op);

  const auto& upstream_output_dims = GetAllValueDimFromValue(
      reduce_out_value, GetUsageIdx(reduce_out_value, downstream_connect_op));
  const auto& [related, _UNUSED] =
      SplitDimsWithRelationship(downstream_reduce_dims, upstream_output_dims);
  auto res = (related.size() == 0);
  VLOG(4) << "ReduceTreeGrownCanMerge: " << res;
  return res;
}

template <typename T>
bool RelativeJudgePolicy<T>::ReducePlusTrivialCanMerge(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  VLOG(4) << "RT can fuse";

  const auto& [upstream_reduce_dims, upstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op_),
                      upstream->sink_op_);

  // usage_idx is not important, for this is downstream output value
  // downstream output value must have been used for there is yield op, so
  // usage_idx==0 exists
  const auto& [_UNUSED, non_related_dims] = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0), 0),
      upstream_non_reduce_dims);

  auto res = DimsEqual(non_related_dims, upstream_reduce_dims) ||
             IsFlattenDimSmaller(upstream, downstream);
  VLOG(4) << "ReducePlusTrivialCanMerge: " << res;
  return res;
}

template <typename T>
bool RelativeJudgePolicy<T>::IsFlattenDimSmaller(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  const auto& [upstream_reduce_dims, upstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op_),
                      upstream->sink_op_);

  const auto& [related_dims, _UNUSED] = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0), 0),
      upstream_non_reduce_dims);

  VLOG(4) << "IsFlattenDimSmaller: "
          << axes_info_.GetSignature(downstream->sink_op_).DebugStr();
  int rank = axes_info_.GetSignature(downstream->sink_op_)
                 .outputs[0]
                 .axis_names.size();
  VLOG(4) << "IsFlattenDimSmaller: " << rank << " " << related_dims.size()
          << " " << upstream_non_reduce_dims.size();
  bool res = (rank - related_dims.size()) <= upstream_non_reduce_dims.size();
  VLOG(4) << "IsFlattenDimSmaller: " << res;
  return res;
}

template <typename T>
bool RelativeJudgePolicy<T>::CanFuse(const PatternNodePtr<T>& upstream,
                                     const PatternNodePtr<T>& downstream) {
  if (std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern_) &&
      std::holds_alternative<TrivialPattern<T>>(downstream->stmt_pattern_)) {
    return ReducePlusTrivialCanMerge(upstream, downstream);
  }
  if (std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern_) &&
      std::holds_alternative<ReduceTreePattern<T>>(downstream->stmt_pattern_)) {
    return ReduceTreeGrownCanMerge(upstream, downstream);
  }
  return true;  // other case.
}

template <typename T>
std::vector<size_t> RelativeJudgePolicy<T>::GetFakeReduceIterIdx(
    const PatternNodePtr<T>& upstream, const PatternNodePtr<T>& downstream) {
  if (!std::holds_alternative<ReduceTreePattern<T>>(upstream->stmt_pattern_) &&
      !std::holds_alternative<TrivialPattern<T>>(downstream->stmt_pattern_)) {
    PADDLE_THROW("Illegal Call GetFakeReduceIterIdx");
  }

  const auto& [upstream_reduce_dims, upstream_non_reduce_dims] =
      SplitReduceDims(axes_info_.GetSignature(upstream->sink_op_),
                      upstream->sink_op_);

  const auto& [_UNUSED, trivial_reorder_dims] = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0), 0),
      upstream_non_reduce_dims);

  // CHECK(upstream_reduce_dims.size() == trivial_reorder_dims.size() ||
  // trivial_reorder_dims.size() == 0);
  std::unordered_set<ValueDim, ValueDimHash> visited_dims;
  std::vector<size_t> result;
  for (auto& reduce_dim : upstream_reduce_dims) {
    for (auto& trivial_dim : trivial_reorder_dims) {
      if (visited_dims.find(trivial_dim) == visited_dims.end() &&
          trivial_dim.GetNumericValue() == reduce_dim.GetNumericValue()) {
        visited_dims.emplace(trivial_dim);
        result.emplace_back(trivial_dim.idx_);
        break;
      }
    }
  }
  VLOG(4) << "FakeReduceIterIdx: " << cinn::utils::Join(result, ", ");
  return result;
}

template class RelativeJudgePolicy<FrontendStage>;
template class RelativeJudgePolicy<BackendStage>;

}  // namespace cinn::fusion
