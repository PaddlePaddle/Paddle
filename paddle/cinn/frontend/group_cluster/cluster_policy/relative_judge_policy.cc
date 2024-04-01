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

SplitDims SplitReduceInputDimsIfRelatedWithNonReduceAxis(
    const ShardableAxesSignature& signature, pir::Operation* op) {
  const auto& v = op->operand_source(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());
  auto result = SplitDims();
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

SplitDims SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
    const ShardableAxesSignature& signature, const pir::Operation* op) {
  const auto& v = op->result(0);
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> input_names_set(input_names.begin(), input_names.end());
  auto result = SplitDims();
  int idx = 0;
  for (const auto& name : output_names) {
    if (input_names_set.count(name) == 0) {
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
  VLOG(4) << "IsBroadcastEdge: upstream_out_dims.size()"
          << upstream_out_dims.size();
  VLOG(4) << "IsBroadcastEdge: downstream_reduce_dims.size()"
          << downstream_reduce_dims.size();

  for (const auto& downstream_reduce_dim : downstream_reduce_dims) {
    for (const auto& upstream_out_dim : upstream_out_dims) {
      VLOG(4) << "upstream_out_dim: " << upstream_out_dim.DebugStr()
              << " downstream_reduce_dim: " << downstream_reduce_dim.DebugStr();
      if (IsRelated(upstream_out_dim, downstream_reduce_dim)) {
        return false;
      }
    }
  }

  VLOG(4) << "IsBroadcastEdge";
  return true;
}

bool RelativeJudgePolicy::ReduceTreeGrownCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& upstream_tree =
      std::get<ReduceTreePattern>(upstream->stmt_pattern_);
  VLOG(4) << "upstream->stmt_pattern_:"
          << OpsDebugStr(GetOpsInPattern(upstream_tree));
  const auto& downstream_tree =
      std::get<ReduceTreePattern>(downstream->stmt_pattern_);
  VLOG(4) << "downstream->stmt_pattern_"
          << OpsDebugStr(GetOpsInPattern(downstream_tree));
  const auto& maybe_downstream_op = GetDownstreamFromCandidate(
      upstream_tree.GetRootPattern(), downstream_tree.reduce_patterns_);
  int idx = 0;
  for (const auto& r_pattern : downstream_tree.reduce_patterns_) {
    idx += 1;
    VLOG(4) << "downstream_tree.reduce_patterns_"
            << "[" << idx << "]" << OpsDebugStr(GetOpsInPattern(r_pattern));
  }
  if (!maybe_downstream_op.has_value()) {
    VLOG(4) << "can't find candidate from patterns. can fuse return false.";
    return false;
  }
  const pir::Value& reduce_out_value =
      upstream_tree.GetRootPattern().GetReduceOp()->result(0);
  pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();
  const auto& split_reduce_dim_result =
      SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(downstream_reduce_op), downstream_reduce_op);
  const auto& upstream_output_dims = GetAllValueDimFromValue(reduce_out_value);
  auto res = IsBroadcastEdge(upstream_output_dims,
                             split_reduce_dim_result.non_related);
  VLOG(4) << "ReduceTreeGrownCanMerge: " << res;
  return res;
}

SplitDims RelativeJudgePolicy::SplitDimsWithRelationship(
    const std::vector<ValueDim>& targets,
    const std::vector<ValueDim>& related_with) {
  VLOG(4) << "SplitDimsWithRelationship";
  auto result = SplitDims();
  bool is_related = false;
  for (auto& target_dim : targets) {
    is_related = false;
    for (auto& related_dim : related_with) {
      if (IsRelated(related_dim, target_dim)) is_related = true;
    }
    if (is_related) {
      result.related.push_back(target_dim);
    } else {
      result.non_related.push_back(target_dim);
    }
  }

  return result;
}

bool DimsEqual(const std::vector<ValueDim>& first,
               const std::vector<ValueDim>& second) {
  VLOG(4) << "DimsEqual";
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

bool RelativeJudgePolicy::ReducePlusTrivialCanMerge(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  VLOG(4) << "RT can fuse";

  // const auto& split_reduce_dims_result =
  //     SplitReduceInputDimsIfRelatedWithNonReduceAxis(
  //         axes_info_.GetSignature(upstream->sink_op_), upstream->sink_op_);

  // VLOG(4) << split_reduce_dims_result.DebugStr();

  // const auto& upstream_reduce_dims = split_reduce_dims_result.non_related;
  // const auto& upstream_non_reduce_dims = split_reduce_dims_result.related;

  // TODO(wuzhanfei) fix bug in relation that if has multi path in graph
  // test_rms_norm can test
  const auto& split_reduce_input_dims_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op_), upstream->sink_op_);
  VLOG(4) << split_reduce_input_dims_result.DebugStr();
  const auto& upstream_reduce_dims = split_reduce_input_dims_result.non_related;

  const auto& split_reduce_output_dims_result =
      SplitReduceOutputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op_), upstream->sink_op_);
  VLOG(4) << split_reduce_input_dims_result.DebugStr();
  const auto& upstream_non_reduce_dims =
      split_reduce_output_dims_result.related;
  // replace codes upside with original design

  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0)),
      upstream_non_reduce_dims);

  VLOG(4) << split_trivial_dims_result.DebugStr();

  auto res =
      DimsEqual(split_trivial_dims_result.non_related, upstream_reduce_dims);
  res = res || IsFlattenDimSmaller(upstream, downstream);
  VLOG(4) << "ReducePlusTrivialCanMerge: " << res;
  return res;
}

bool RelativeJudgePolicy::IsFlattenDimSmaller(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  const auto& split_reduce_dims_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op_), upstream->sink_op_);
  const auto& upstream_reduce_dims = split_reduce_dims_result.non_related;
  const auto& upstream_non_reduce_dims = split_reduce_dims_result.related;

  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0)),
      upstream_non_reduce_dims);

  VLOG(4) << "IsFlattenDimSmaller: "
          << axes_info_.GetSignature(downstream->sink_op_).DebugStr();
  int rank = axes_info_.GetSignature(downstream->sink_op_)
                 .outputs[0]
                 .axis_names.size();
  VLOG(4) << "IsFlattenDimSmaller: " << rank << " "
          << split_trivial_dims_result.related.size() << " "
          << upstream_non_reduce_dims.size();
  bool res = (rank - split_trivial_dims_result.related.size()) <=
             upstream_non_reduce_dims.size();
  VLOG(4) << "IsFlattenDimSmaller: " << res;
  return res;
}

bool RelativeJudgePolicy::CanFuse(const PatternNodePtr& upstream,
                                  const PatternNodePtr& downstream) {
  if (upstream->IsReduceTree() || downstream->IsTrivial()) {
    return ReducePlusTrivialCanMerge(upstream, downstream);
  }
  if (upstream->IsReduceTree() || downstream->IsReduceTree()) {
    return ReduceTreeGrownCanMerge(upstream, downstream);
  }
  return true;  // other case.
}

std::vector<size_t> RelativeJudgePolicy::GetFakeReduceIterIdx(
    const PatternNodePtr& upstream, const PatternNodePtr& downstream) {
  if (!upstream->IsReduceTree() || !downstream->IsTrivial()) {
    PADDLE_THROW("Illegal Call GetFakeReduceIterIdx");
  }

  const auto& split_reduce_dims_result =
      SplitReduceInputDimsIfRelatedWithNonReduceAxis(
          axes_info_.GetSignature(upstream->sink_op_), upstream->sink_op_);

  const auto& upstream_reduce_dims = split_reduce_dims_result.non_related;
  const auto& upstream_non_reduce_dims = split_reduce_dims_result.related;

  const auto& split_trivial_dims_result = SplitDimsWithRelationship(
      GetAllValueDimFromValue(downstream->sink_op_->result(0)),
      upstream_non_reduce_dims);

  const auto& trivial_reorder_dims = split_trivial_dims_result.non_related;

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

}  // namespace cinn::frontend::group_cluster::policy
