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

#include "glog/logging.h"

#include "paddle/cinn/operator_fusion/backend/pattern.h"
#include "paddle/cinn/operator_fusion/backend/pattern_api.h"
#include "paddle/cinn/hlir/framework/pir/trivial_op_impl.h"

namespace cinn::fusion {

template <>
StmtPattern<BackendStage> ConvertToStmtPattern(const PatternContent<BackendStage>& content) {
  const auto& kind = GetOpPatternKind(content.op);
  if (kind == hlir::framework::kReduction) {
    CHECK(content.expr.has_value());
    return ReducePattern<BackendStage>({content.op}, ReduceOp(content.expr.value()));
  } else if (kind == hlir::framework::kElementWise ||
             kind == hlir::framework::kBroadcast ||
             kind == hlir::framework::kInjective) {
    CHECK(content.expr.has_value());
    return TrivialPattern<BackendStage>({content.op}, TrivialOp(content.expr.value()));
  } else {
    CHECK(false);
    return UnsupportPattern<BackendStage>({content.op});
  }
}

template <>
StmtPattern<BackendStage> RT_x_RT(const ReduceTreePattern<BackendStage>& first,
                                  const ReduceTreePattern<BackendStage>& second) {
    const auto& merged = ConcatVector(first.reduce_patterns(),
                                      second.reduce_patterns());
    return ReduceTreePattern<BackendStage>(merged, second.GetRootPattern());
}

template <>
StmtPattern<BackendStage> RT_x_Trivial(const ReduceTreePattern<BackendStage>& first,
                                       const TrivialPattern<BackendStage>& second) {
  return ReduceTreePlusTrivialPattern<BackendStage>(first, second);
}

template <>
StmtPattern<BackendStage> Trivial_x_Reduce(const TrivialPattern<BackendStage>& first,
                                           const ReducePattern<BackendStage>& second) {
  const auto& ops =
      MergeVector(GetOpsInPattern<BackendStage>(first), GetOpsInPattern<BackendStage>(second));
  const auto& reduce_op = cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(first.trivial_op, second.reduce_op);
  return ReducePattern<BackendStage>(ops, reduce_op);
}

template <>
StmtPattern<BackendStage> Trivial_x_Trivial(const TrivialPattern<BackendStage>& first,
                                            const TrivialPattern<BackendStage>& second) {
  const auto& ops =
      MergeVector(GetOpsInPattern<BackendStage>(first), GetOpsInPattern<BackendStage>(second));
  const auto& trivial_op = cinn::hlir::framework::pir::trivial_fusion_detail::TrivalxOther_Fusion(first.trivial_op, second.trivial_op);
  return TrivialPattern<BackendStage>(ops, trivial_op);
}

template <>
StmtPattern<BackendStage> H_x_H(const HorizontalFusionPattern<BackendStage>& first,
                     const HorizontalFusionPattern<BackendStage>& second) {
  const auto& contents =
      MergeVector(GetOpsInPattern<BackendStage>(first), GetOpsInPattern<BackendStage>(second));
  return HorizontalFusionPattern<BackendStage>({first, second});
}

struct FusionOpGetter {
  std::vector<FusionOp> operator()(const TrivialPattern<BackendStage>& pattern) {
    return {pattern.trivial_op};
  }
  std::vector<FusionOp> operator()(const ReducePattern<BackendStage>& pattern) {
    return {pattern.reduce_op};
  }
  // TODO: add tmp transform in this function.
  std::vector<FusionOp> operator()(const ReduceTreePattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
  std::vector<FusionOp> operator()(const ReduceTreePlusTrivialPattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
  std::vector<FusionOp> operator()(const HorizontalFusionPattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
  std::vector<FusionOp> operator()(const UnsupportPattern<BackendStage>& pattern) {
    CHECK(false) << "Not Implemented.";
  }
};

// tmp transform for reduce_tree and reduce_tree_trivial.
std::vector<FusionOp> GetFusionOpFromPattern (const StmtPattern<BackendStage>& pattern) {
  return std::visit(FusionOpGetter(), pattern);
}

struct FusionOp2Expr {
  std::vector<ir::Expr> operator()(const TrivialOp& op) {
    return {op.GetFuncBody()};
  }
  std::vector<ir::Expr> operator()(const ReduceOp& op) {
    const auto& t_r = SplitReduceOp(op);
    return {t_r.first.GetFuncBody(), t_r.second.GetFuncBody()};
  }
};

std::vector<ir::Expr> GetExprFromPattern (const StmtPattern<BackendStage>& pattern) {
  const auto& fusion_ops = GetFusionOpFromPattern(pattern);
  std::vector<ir::Expr> results;
  for (const auto& op : fusion_ops) {
    results = ConcatVector(results, std::visit(FusionOp2Expr(), op));
  }
  return results;
}

}
