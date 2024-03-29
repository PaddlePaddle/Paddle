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

  size_t GetNumbericValue() const {
    return v_.type().dyn_cast<pir::DenseTensorType>().dims().at(idx_);
  }

  std::string DebugStr() const {
    std::ostringstream oss;
    oss << "ValueDim: ";
    oss << "Index: " << idx_;
    oss << ", ";
    v_.defining_op()->Print(oss);
    return oss.str();
  }
};

struct ValueDimHash {
  std::size_t operator()(const ValueDim& p) const {
    auto h1 = std::hash<size_t>{}(p.idx_);
    auto h2 = std::hash<pir::Value>{}(p.v_);
    // Mainly for demonstration purposes, i.e. works but is overly simple
    // In the real world, use sth. like boost.hash_combine
    return h1 ^ (h2 << 1);
  }
};

using ValueDimRelation =
    std::unordered_map<ValueDim,
                       std::unordered_map<ValueDim, bool, ValueDimHash>,
                       ValueDimHash>;
// ValueDimRelation[in][out] = True; means f(out) = in is related.

static std::vector<ValueDim> GetAllValueDimFromValue(const pir::Value& v) {
  std::vector<ValueDim> value_dims;
  size_t rank = GetRank(v);
  for (size_t i = 0; i < rank; ++i) {
    value_dims.emplace_back(v, i);
  }
  return value_dims;
}

static std::vector<ValueDim> GetAllInputValueDim(pir::Operation* op) {
  std::vector<ValueDim> value_dims;
  for (const auto& v : op->operands()) {
    value_dims = ConcatVector(value_dims, GetAllValueDimFromValue(v.source()));
  }
  return value_dims;
}

static std::vector<ValueDim> GetAllOutputValueDim(pir::Operation* op) {
  std::vector<ValueDim> value_dims;
  for (const auto& v : op->results()) {
    value_dims = ConcatVector(value_dims, GetAllValueDimFromValue(v));
  }
  return value_dims;
}

static ValueDimRelation CreateOpRelativenessForElementWise(pir::Operation* op) {
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

static std::vector<std::pair<size_t, size_t>> GetNonBroadCastDims(
    pir::Operation* op) {
  // TODO(xk): only static shape here!
  std::vector<std::pair<size_t, size_t>> res;
  if (op->name() == "cinn_op.broadcast") {
    const auto& in_dim =
        op->operand(0).type().dyn_cast<pir::DenseTensorType>().dims();
    const auto& out_dim =
        op->result(0).type().dyn_cast<pir::DenseTensorType>().dims();
    // CINN_CHECK_EQ(in_dim.size(), out_dim.size());
    for (int i = 1; i <= in_dim.size(); ++i) {
      if (in_dim.size() - i < 0 || out_dim.size() - i < 0) break;
      if (in_dim[in_dim.size() - i] == out_dim[out_dim.size() - i]) {
        res.emplace_back(in_dim.size() - i, out_dim.size() - i);
      }
    }
  } else if (op->name() == "pd_op.expand") {
    auto* mut_op = const_cast<pir::Operation*>(op);
    auto expand_op = mut_op->dyn_cast<paddle::dialect::ExpandOp>();

    const auto& input_value = expand_op.x();
    const auto& output_value = expand_op.out();

    const int input_rank = GetRank(input_value);
    const int output_rank = GetRank(output_value);
    // CHECK_GE(output_rank, input_rank);

    // TODO(Baizhou): How to fetch shape_analysis in a more elegant way
    const auto* shape_analysis =
        &pir::ShapeAnalysisManager::Instance().Get(op->GetParentProgram());

    for (int i = 1; i <= input_rank; ++i) {
      if (input_rank - i < 0 || output_rank - i < 0) break;
      if (shape_analysis->IsProductEqual(
              input_value, {input_rank - i}, output_value, {output_rank - i})) {
        res.emplace_back(input_rank - i, output_rank - i);
      }
    }
  } else {
    CHECK(false) << "Not Implement other broadcast op.";
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForBroadcast(pir::Operation* op) {
  ValueDimRelation res;
  const auto& in_value = op->operand(0).source();
  const auto& out_value = op->result(0);
  for (const auto& t : GetNonBroadCastDims(op)) {
    res[ValueDim(in_value, t.first)][ValueDim(out_value, t.second)] = true;
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForDefault(pir::Operation* op) {
  ValueDimRelation res;
  for (const auto& out_dim : GetAllOutputValueDim(op)) {
    for (const auto& in_dim : GetAllInputValueDim(op)) {
      res[in_dim][out_dim] = true;
    }
  }
  return res;
}

static ValueDimRelation CreateOpRelativenessForReduce(pir::Operation* op) {
  const auto& reduce_axis_idx = GetReduceAxisIdx(op);
  ValueDimRelation res;
  const size_t input_rank = GetRank(op->operand_source(0));
  int out_idx = 0;
  bool keep_dim = GetReduceOpKeepDims(op);
  for (int i = 0; i < input_rank; i++) {
    if (std::find(reduce_axis_idx.begin(), reduce_axis_idx.end(), i) !=
        reduce_axis_idx.end()) {
      res[ValueDim(op->operand_source(0), i)]
         [ValueDim(op->result(0), out_idx)] = true;
      out_idx += 1;
    } else {
      out_idx += keep_dim;
    }
  }
  return res;
}

static std::optional<ValueDimRelation> CreateOpRelativenessForSpecialOps(
    pir::Operation* op) {
  if (op->name() == "cinn_op.reshape") {
    // Special Elementwise.
    return CreateOpRelativenessForDefault(op);
  }
  return {};
}

static ValueDimRelation GetSingleOpRelation(pir::Operation* op) {
  VLOG(4) << "GetSingleOpRelation for " << op->name();
  const auto& special_result = CreateOpRelativenessForSpecialOps(op);
  if (special_result != std::nullopt) {
    return special_result.value();
  }

  CHECK(op->num_results() == 1)
      << "Now we do not support op with multi outputs: " << op->name();
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  ValueDimRelation result;
  if (kind == hlir::framework::kReduction) {
    result = CreateOpRelativenessForReduce(op);
  } else if (kind == hlir::framework::kElementWise) {
    result = CreateOpRelativenessForElementWise(op);
  } else if (kind == hlir::framework::kBroadcast) {
    result = CreateOpRelativenessForBroadcast(op);
  } else {
    result = CreateOpRelativenessForDefault(op);
  }
  return result;
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
    const std::vector<pir::Operation*>& ops) {
  ValueDimRelation res;

  for (size_t i = ops.size(); i >= 1; --i) {
    pir::Operation* op = ops[i - 1];
    if (op->name() == "cf.yield") continue;

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

struct SplitedDims {
  std::vector<ValueDim> related;
  std::vector<ValueDim> non_related;

  std::string DebugStr() const {
    std::stringstream ss;
    ss << "SplitedDims:\nrelated:\n";
    for (const auto& dim : related) {
      ss << dim.DebugStr() << "\n";
    }
    ss << "non_related:\n";
    for (const auto& dim : non_related) {
      ss << dim.DebugStr() << "\n";
    }
    return ss.str();
  }
};

class RelativeJudgePolicy final : public Policy {
 public:
  RelativeJudgePolicy(const std::vector<pir::Operation*>& ops,
                      const pir::ShapeConstraintIRAnalysis* shape_analysis)
      : axes_info_(ops, shape_analysis) {
    VLOG(4) << "[relative_judge_policy] Start AnalysisIndexExprRelation.";
    index_expr_map_ = AnalysisIndexExprRelation(ops);
    VLOG(4) << "[relative_judge_policy] End AnalysisIndexExprRelation.";
  }
  bool CanFuse(const PatternNodePtr& upstream,
               const PatternNodePtr& downstream) override;

  std::string Name() { return "RelativeJudgePolicy"; }

  std::vector<size_t> GetFakeReduceIterIdx(
      const PatternNodePtr& upstream,
      const PatternNodePtr& downstream) override;

  bool IsRelated(ValueDim in, ValueDim out) {
    return index_expr_map_[in].count(out) == 1;
  }

 private:
  ValueDimRelation index_expr_map_;
  ShardableAxesInfoManager axes_info_;
  bool ReduceTreeGrownCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  bool IsFlattenDimSmaller(const PatternNodePtr& upstream,
                           const PatternNodePtr& downstream);
  bool ReducePlusTrivialCanMerge(const PatternNodePtr&, const PatternNodePtr&);
  SplitedDims SplitDimsWithRelationship(
      const std::vector<ValueDim>& targets,
      const std::vector<ValueDim>& related_with);
  std::optional<ReducePattern> GetDownstreamFromCandidate(
      const ReducePattern& upstream,
      const std::vector<ReducePattern>& candidates);
  bool IsDownstreamStmtDependReduceOp(pir::Operation* reduce,
                                      const StmtPattern& downstream);
  bool IsBroadcastEdge(const std::vector<ValueDim>& upstream_out_dims,
                       const std::vector<ValueDim>&);
};

}  // namespace cinn::frontend::group_cluster::policy
