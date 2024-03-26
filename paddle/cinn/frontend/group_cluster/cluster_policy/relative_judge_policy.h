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

#pragma once
#include <functional>
#include "paddle/cinn/frontend/group_cluster/cluster_policy/policy_manager.h"
#include "paddle/cinn/frontend/group_cluster/cluster_policy/shardable_axes_policy/shardable_axes_base.h"
#include "paddle/cinn/frontend/group_cluster/common_utils.h"

namespace cinn::frontend::group_cluster::policy {

struct ValueDim {
  pir::Value v_;
  size_t idx_;
  ValueDim(pir::Value v, size_t idx) : v_(v), idx_(idx) {}
  ValueDim() = default;
  ValueDim(const ValueDim& v) = default;
  bool operator==(const ValueDim& v) const {
    return (idx_ == v.idx_) && (v_ == v.v_);
  }
};

struct ValueDimHash {
  std::size_t operator()(const ValueDim& p) const {
    auto h1 = std::hash<size_t>{}(p.idx_);
    auto h2 = std::hash<pir::Value>{}(p.v_);
    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ h2;
  }
};

using ValueDimRelation =
    std::unordered_map<ValueDim,
                       std::unordered_map<ValueDim, bool, ValueDimHash>,
                       ValueDimHash>;
// ValueDimRelation[in][out] = True; means f(out) = in is related.

static std::optional<ValueDimRelation> CreateOpRelativenessForSpecialOps(
    const pir::Operation* op) {
  return {};
}

static std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v) {
  std::vector<ValueDim> value_dims;
  size_t rank = GetRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dims.emplace_back(v, i);
  }
  return value_dims;
}

static std::vector<ValueDim> GetAllInputValueDim(const pir::Operation* op) {
  std::vector<ValueDim> value_dims;
  for (const auto& v : op->operands()) {
    value_dims = ConcatVector(value_dims, GetAllValueDimFromValue(v.source()));
  }
  return value_dims;
}

static std::vector<ValueDim> GetAllOutputValueDim(const pir::Operation* op) {
  std::vector<ValueDim> value_dims;
  for (const auto& v : op->results()) {
    value_dims = ConcatVector(value_dims, GetAllValueDimFromValue(v));
  }
  return value_dims;
}

static ValueDimRelation CreateOpRelativenessForElementWise(
    const pir::Operation* op) {
  ValueDimRelation res;
  for (const auto& v : op->operands()) {
    const auto& value_dims = GetAllValueDimFromValue(v.source());
    const auto& out_value_dims = GetAllOutputValueDim(op);
    CHECK_EQ(value_dims.size(), out_value_dims.size());
    for (size_t i = 0; i < value_dims.size(); ++i) {
      res[value_dims[i]][out_value_dims[i]] = true;
    }
  }
  return res;
}

static std::vector<size_t> GetNonBroadCastDims(const pir::Operation* op) {
  // TODO: only static shape here!
  std::vector<size_t> res;
  if (op->name() == "cinn_op.broadcast") {
    const auto& in_dim =
        op->operand(0).type().dyn_cast<pir::DenseTensorType>().dims();
    const auto& out_dim =
        op->result(0).type().dyn_cast<pir::DenseTensorType>().dims();
    CINN_CHECK_EQ(in_dim.size(), out_dim.size());
    for (int i = 0; i < in_dim.size(); ++i) {
      if (in_dim[i] == out_dim[i]) {
        res.push_back(i);
      }
    }
  } else {
    // TODO: not implemented.
    CINN_CHECK(false);
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForBroadcast(
    const pir::Operation* op) {
  ValueDimRelation res;
  const auto& in_value = op->operand(0).source();
  const auto& out_value = op->result(0);
  for (size_t t : GetNonBroadCastDims(op)) {
    res[ValueDim(in_value, t)][ValueDim(out_value, t)] = true;
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForDefault(
    const pir::Operation* op) {
  ValueDimRelation res;
  for (const auto& out_dim : GetAllOutputValueDim(op)) {
    for (const auto& in_dim : GetAllInputValueDim(op)) {
      res[in_dim][out_dim] = true;
    }
  }
  return res;
}

static ValueDimRelation GetSingleOpRelation(const pir::Operation* op) {
  auto special_result = CreateOpRelativenessForSpecialOps(op);
  if (special_result != std::nullopt) {
    return special_result.value();
  }

  CHECK(op->num_results() == 1)
      << "Now we do not support op with multi outputs";
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  ValueDimRelation result;
  if (kind == hlir::framework::kElementWise) {
    result = CreateOpRelativenessForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateOpRelativenessForBroadcast(op);
  } else {
    result = CreateOpRelativenessForDefault(op);
  }
  // VLOG(4) << "[relative_judge_policy] Create OpDimRelativeness : \n"
  //<< op->name() << " : " << result.DebugStr();
}

static std::vector<std::pair<ValueDim, ValueDim>> FlattenRelation(
    const ValueDimRelation& axes_relation) {
  std::vector<std::pair<ValueDim, ValueDim>> res;
  for (const auto& in_dim_pair : axes_relation) {
    for (const auto& out_dim_pair : in_dim_pair.second) {
      res.emplace_back(in_dim_pair.first, out_dim_pair.first);
    }
  }
  return res;
}

static ValueDimRelation AnalysisIndexExprRelation(
    const std::vector<const pir::Operation*>& ops) {
  ValueDimRelation res;
  for (size_t i = ops.size() - 1; i >= 0; --i) {
    const pir::Operation* op = ops[i];
    const auto& value_dim_relation = GetSingleOpRelation(op);
    for (const auto& in_out_pair : FlattenRelation(value_dim_relation)) {
      for (const auto& out_relation : res[in_out_pair.second]) {
        res[in_out_pair.first][out_relation.first] = true;
      }
      res[in_out_pair.first][in_out_pair.second] = true;
    }
  }
  return res;
}

class RelativeJudgePolicy final : public Policy {
 public:
  RelativeJudgePolicy(const std::vector<const pir::Operation*>& ops,
                      const pir::ShapeConstraintIRAnalysis* shape_analysis)
      : axes_info_(ops, shape_analysis) {
    index_expr_map_ = AnalysisIndexExprRelation(ops);
  }
  bool CanFuse(const PatternNodePtr& upstream,
               const PatternNodePtr& downstream) override;
  bool IsRelated(ValueDim in, ValueDim out) {
    return index_expr_map_[in].count(out) == 1;
  }

 private:
  ValueDimRelation index_expr_map_;
  ShardableAxesInfoManager axes_info_;
  bool ReduceTreeGrownCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  std::optional<ReducePattern> GetDownstreamFromCandidate(
      const ReducePattern& upstream,
      const std::vector<ReducePattern>& candidates);
  bool IsDownstreamStmtDependReduceOp(const pir::Operation* reduce,
                                      const StmtPattern& downstream);
  bool IsBroadcastEdge(const std::vector<ValueDim>& upstream_out_dims,
                       const std::vector<ValueDim>&);
};

}  // namespace cinn::frontend::group_cluster::policy
