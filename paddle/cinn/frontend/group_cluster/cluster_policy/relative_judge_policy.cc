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
    const pir::Operation* reduce, const StmtPattern& downstream) {
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
  const pir::Operation* reduce = upstream.GetReduceOp();
  for (const auto& candidate : candidates) {
    if (IsDownstreamStmtDependReduceOp(reduce, candidate)) {
      return candidate;
    }
  }
  return {};
}

inline static std::vector<ValueDim> GetReduceAxesValueDims(
    const ShardableAxesSignature& signature, const pir::Value& v) {
  const auto& input_names = signature.inputs[0].axis_names;
  const auto& output_names = signature.outputs[0].axis_names;
  std::set<std::string> output_names_set(output_names.begin(),
                                         output_names.end());
  std::vector<ValueDim> res;
  int idx = 0;
  for (const auto& in : input_names) {
    if (output_names_set.count(in) == 0) {
      res.emplace_back(v, idx);
    }
    idx += 1;
  }
  return res;
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
  const pir::Operation* downstream_reduce_op =
      maybe_downstream_op.value().GetReduceOp();
  VLOG(4) << "downstream_reduce_op: " << OpsDebugStr({downstream_reduce_op});
  const auto& reduce_value_dims =
      GetReduceAxesValueDims(axes_info_.GetSignature(downstream_reduce_op),
                             downstream_reduce_op->result(0));
  const auto& upstream_output_dims = GetAllValueDimFromValue(reduce_out_value);
  return IsBroadcastEdge(upstream_output_dims, reduce_value_dims);
}

bool RelativeJudgePolicy::CanFuse(const PatternNodePtr& upstream,
                                  const PatternNodePtr& downstream) {
  if (!upstream->IsReduceTree() || !downstream->IsReduceTree()) {
    return true;
  }
  return ReduceTreeGrownCanMerge(upstream, downstream);
}
}  // namespace cinn::frontend::group_cluster::policy
