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

#include "paddle/cinn/frontend/cluster_ops/fusion_helper.h"

namespace cinn::frontend::cluster_ops {

struct FusePolicy_IS_x_PS_2_PS {
  static bool FuseCondition(const StmtPattern& upstream,
                            const StmtPattern& downstream) {
    return IsISPattern(upstream) && IsPSPattern(downstream);
  }

  static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const StmtPattern& upstream, const StmtPattern& downstream) {
    return MergePatternImpl(std::get<IS>(upstream), std::get<PS>(downstream));
  }

  static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
      const IS& upstream, const PS& downstream) {
    const auto& ops = [&] {
      std::vector<const pir::Operation*> ops(upstream.ops.begin(),
                                              upstream.ops.end());
      for (const auto* downstream_op : downstream.ops) {
        if (std::find(ops.begin(), ops.end(), downstream_op) == ops.end()) {
          ops.push_back(downstream_op);
        }
      }
      return ops;
    }();
    const auto& shardable_axes_signature =
        MergeShardableAxesSignature(upstream, downstream);
    return StmtPattern(PS{
        .ops = ops,
        .sole_sink = downstream.sole_sink,
        .shardable_axes_signature = shardable_axes_signature,
    });
  }

  ShardableAxesSignature MergeShardableAxesSignature(
      const IS& upstream, const PS& downstream) {
    LOG(FATAL) << "TODO(tianchao)";
  }
};


struct FusePolicy_IS_x_R_2_R {
  static bool FuseCondition(const StmtPattern& upstream,
                            const StmtPattern& downstream) {
    return IsISPattern(upstream) && IsRPattern(downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
        const StmtPattern& upstream, const StmtPattern& downstream);
    std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
        const IS& upstream, const R& downstream);
  };

  static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const StmtPattern& upstream, const StmtPattern& downstream) {
    return MergePatternImpl(std::get<IS>(upstream), std::get<R>(downstream));
  }

  static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
      const IS& upstream, const R& downstream) {
    if (downstream.HasFusedInput()) {
      return ErrorGroupPattern{
          .ops = {downstream.reduce_op_pattern.reduce_op},
          .error_string = "The input of reduce has been fused.",
      };
    }
    R new_pattern = R(downstream);
    new_pattern.input = upstream;
    return StmtPattern(std::move(new_pattern));
  }
};


struct FusePolicy_PS_x_R_2_R {
  static bool FuseCondition(const StmtPattern& upstream,
                            const StmtPattern& downstream) {
    return IsISPattern(upstream) && IsRPattern(downstream);
  }

  static std::variant<StmtPattern, ErrorGroupPattern> MergePattern(
      const StmtPattern& upstream, const StmtPattern& downstream) {
    return MergePatternImpl(std::get<PS>(upstream), std::get<R>(downstream));
  }

  static std::variant<StmtPattern, ErrorGroupPattern> MergePatternImpl(
      const PS& upstream, const R& downstream) {
    if (downstream.HasFusedInput()) {
      return ErrorGroupPattern{
          .ops = {downstream.reduce_op_pattern.reduce_op},
          .error_string = "The input of reduce has been fused.",
      };
    }
    R new_pattern = R(downstream);
    new_pattern.input = upstream;
    return StmtPattern(new_pattern);
  }
};

StmtFusionHelper::StmtFusionHelper(
    const std::vector<const pir::Operation*>& ops,
    const ShardableAxesInferer& shardable_axes_inferer)
    : ops_(ops), shardable_axes_inferer_(shardable_axes_inferer) {
  this->op_topo_ = OpTopo::Make(ops);
  this->IsInThisOpList = MakePredicatorIsInThisFusionOp(ops);
  this->IsInjectiveSource = MakePredicatorIsInjectiveSource(this->op_topo_);
  this->GetOrderValue4Op = MakeTopoOrderFinderOfOp(ops);
}

GroupPattern StmtFusionHelper::FuseToGroupPattern() {
  std::vector<StmtPattern> stmt_patterns = ConvertToStmtPatternVec();
  if (const auto& error = Fuse_IS_x_IS_2_IS(&stmt_patterns))
    return error.value();
  if (const auto& error = Fuse_PS_x_PS_2_PS(&stmt_patterns))
    return error.value();
  if (const auto& error = Fuse_IS_x_PS_2_PS(&stmt_patterns))
    return error.value();
  if (const auto& error = Fuse_IS_x_R_2_R(&stmt_patterns)) return error.value();
  if (const auto& error = Fuse_PS_x_R_2_R(&stmt_patterns)) return error.value();
  SortStmtPatterns(&stmt_patterns);
  return stmt_patterns;
}

std::vector<StmtPattern> StmtFusionHelper::ConvertToStmtPatternVec() {
  std::vector<StmtPattern> ret;
  for (const auto* op : ops_) {
    if (!IsInThisOpList(op)) continue;
    ret.emplace_back(ConvertToStmtPattern(op));
  }
  return ret;
}

void StmtFusionHelper::SortStmtPatterns(
    std::vector<StmtPattern>* stmt_patterns) {
  std::vector<const StmtPattern*> stmt_ptr_patterns = [&] {
    std::vector<const StmtPattern*> stmt_ptr_patterns;
    stmt_ptr_patterns.reserve(stmt_patterns->size());
    for (const auto& stmt_pattern : *stmt_patterns) {
      stmt_ptr_patterns.push_back(&stmt_pattern);
    }
    return stmt_ptr_patterns;
  }();
  SortStmtPtrs(&stmt_ptr_patterns, this->GetOrderValue4Op);
  *stmt_patterns = [&] {
    std::vector<StmtPattern> sorted_stmts;
    sorted_stmts.reserve(stmt_ptr_patterns.size());
    for (const auto* stmt_ptr : stmt_ptr_patterns) {
      sorted_stmts.push_back(*stmt_ptr);
    }
    return sorted_stmts;
  }();
}

std::optional<ErrorGroupPattern> StmtFusionHelper::Fuse_IS_x_IS_2_IS(
    std::vector<StmtPattern>* stmt_patterns) {
  const auto ConstructISPattern = [&](const auto& ops) {
    return IS{
        .ops = ops,
        .sole_sink = GetSoleSink(OpSet(ops.begin(), ops.end())),
    };
  };
  return MultiFuse(IsISPattern, ConstructISPattern, stmt_patterns);
}

std::optional<ErrorGroupPattern> StmtFusionHelper::Fuse_PS_x_PS_2_PS(
    std::vector<StmtPattern>* stmt_patterns) {
  const auto ConstructPSPattern = [&](const auto& ops) {
    auto op_topo = OpTopo::Make(ops);
    const auto shardable_axes_signature = GetShardableAxesSignature(op_topo);
    return PS{
        .ops = ops,
        .sole_sink = GetSoleSink(OpSet(ops.begin(), ops.end())),
        .shardable_axes_signature = shardable_axes_signature,
    };
  };
  return MultiFuse(IsPSPattern, ConstructPSPattern, stmt_patterns);
}

std::optional<ErrorGroupPattern> StmtFusionHelper::Fuse_IS_x_PS_2_PS(
    std::vector<StmtPattern>* stmt_patterns) {
  return FuseFilteredStmtPatterns<FusePolicy_IS_x_PS_2_PS>(stmt_patterns);
}

std::optional<ErrorGroupPattern> StmtFusionHelper::Fuse_IS_x_R_2_R(
    std::vector<StmtPattern>* stmt_patterns) {
  return FuseFilteredStmtPatterns<FusePolicy_IS_x_R_2_R>(stmt_patterns);
}

std::optional<ErrorGroupPattern> StmtFusionHelper::Fuse_PS_x_R_2_R(
    std::vector<StmtPattern>* stmt_patterns) {
  return FuseFilteredStmtPatterns<FusePolicy_PS_x_R_2_R>(stmt_patterns);
}

StmtPattern StmtFusionHelper::ConvertToStmtPattern(const pir::Operation* op) {
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  if (IsInjectiveSource(op)) {
    return ConvertToIS(op);
  } else if (kind == hlir::framework::kReduction) {
    return ConvertReductionOpToReductionPattern(op);
  } else if (kind == hlir::framework::kElementWise) {
    return ConvertOpToPS(op);
  } else if (kind == hlir::framework::kBroadcast) {
    return ConvertOpToPS(op);
  } else {
    LOG(FATAL)
        << "only kReduction, kElementWise, kBroadcast supported. op_name:"
        << op->name();
  }
  LOG(FATAL) << "Dead code";
}

IS StmtFusionHelper::ConvertToIS(const pir::Operation* op) {
  VLOG(4) << "Converting Op to IS";
  return IS{
      .ops = {op},
      .sole_sink = op,
  };
}

R StmtFusionHelper::ConvertReductionOpToReductionPattern(
    const pir::Operation* op) {
  VLOG(4) << "Converting Op to R";
  return R{{}, {op}};
}

PS StmtFusionHelper::ConvertOpToPS(const pir::Operation* op) {
  VLOG(4) << "Converting Op to PS";
  const hlir::framework::OpPatternKind kind = GetOpPatternKind(op);
  const auto shardable_axes_signature =
      shardable_axes_inferer_.MakeShardableAxesSignature4Op(op);
  return PS{
      .ops = {op},
      .sole_sink = op,
      .shardable_axes_signature = shardable_axes_signature,
  };
}

StmtFusionHelper::StmtPtr4OpT StmtFusionHelper::MakeStmtFinderFromOp(
    std::vector<StmtPattern>* stmts) {
  std::unordered_map<const pir::Operation*, StmtPattern*> op2stmt_ptr;
  for (auto& stmt : *stmts) {
    VisitStmtOp(stmt, [&](const auto* op) { op2stmt_ptr[op] = &stmt; });
  }
  return [map = std::move(op2stmt_ptr)](
             const pir::Operation* op) -> std::optional<StmtPattern*> {
    const auto iter = map.find(op);
    if (iter == map.end()) return std::nullopt;
    return iter->second;
  };
}

bool StmtFusionHelper::IsConnected(
    const StmtFusionHelper::StmtPtr4OpT& StmtFinder,
    const StmtPattern* upstream,
    const StmtPattern* downstream) {
  const auto VisitInputStmt = [&](const StmtPattern* stmt,
                                  const StmtVisitor& DoEach) {
    VisitStmtOp(*stmt, [&](const auto* op) {
      op_topo_.VisitInputOp(op, [&](const pir::Operation* input) {
        if (const auto& input_stmt = StmtFinder(input)) {
          DoEach(input_stmt.value());
        }
      });
    });
  };

  bool found = false;
  VisitInputStmt(downstream, [&](const StmtPattern* input_pattern) {
    if (input_pattern == upstream) {
      found = true;
    }
  });
  return found;
}

template <typename FusionPolicy>
std::optional<ErrorGroupPattern>  StmtFusionHelper::FuseFilteredStmtPatterns(
    std::vector<StmtPattern>* stmt_patterns) {
  std::list<StmtPattern*> stmts_iters = [&] {
    std::list<StmtPattern*> stmts_iters;
    for (auto& stmt : *stmt_patterns) {
      stmts_iters.push_back(&stmt);
    }
    return stmts_iters;
  }();
  const auto StmtFinder = MakeStmtFinderFromOp(stmt_patterns);
  const auto EraseOld = [&](const StmtIterPair& pattern_pair) {
    stmts_iters.erase(pattern_pair.upstream_iter);
    stmts_iters.erase(pattern_pair.downstream_iter);
  };
  const auto& InsertNew = [&](const StmtPattern& stmt_pattern) {
    stmt_patterns->push_back(stmt_pattern);
    stmts_iters.push_back(&stmt_patterns->back());
  };
  while (true) {
    const auto& pattern_pair = FindConnetedPattenPairWithCondition(
        StmtFinder, &stmts_iters, &FusionPolicy::FuseCondition);
    if (!pattern_pair.has_value()) break;
    const std::variant<StmtPattern, ErrorGroupPattern>& new_pattern =
        FusionPolicy::MergePattern(**pattern_pair.value().upstream_iter,
                                    **pattern_pair.value().downstream_iter);

    if (std::holds_alternative<ErrorGroupPattern>(new_pattern)) {
      return std::get<ErrorGroupPattern>(new_pattern);
    }
    EraseOld(pattern_pair.value());
    InsertNew(std::get<StmtPattern>(new_pattern));
  }
  *stmt_patterns = [&] {
    std::vector<StmtPattern> ret_patterns;
    ret_patterns.reserve(stmts_iters.size());
    for (const auto& stmt_iter : stmts_iters) {
      ret_patterns.push_back(*stmt_iter);
    }
    return ret_patterns;
  }();
  return std::nullopt;
}

ShardableAxesSignature StmtFusionHelper::GetShardableAxesSignature(
    const OpTopo& op_topo) {
  const pir::Operation* sink = [&] {
    const auto& sinks = GetSinks(*op_topo.ops);
    CHECK_EQ(sinks.size(), 1) << "ops must have only one sink node.";
    return *sinks.begin();
  }();
  const auto& value2shardable_axes =
      shardable_axes_inferer_.InferShardableAxesFromSink(sink, op_topo);
  const auto& IsInputOpOperand = [&](const auto* op, int input_idx) {
    const auto& defining_op = op->operand_source(input_idx).defining_op();
    return IsInThisOpList(defining_op) && op_topo.ops->count(defining_op) == 0;
  };
  const auto& input_op_operands = [&] {
    std::vector<OpAndOperandIndex> op_operands;
    for (const auto* op : *op_topo.ops) {
      for (int i = 0; i < op->num_operands(); ++i) {
        if (!IsInputOpOperand(op, i)) continue;
        op_operands.emplace_back(OpAndOperandIndex{op, i});
      }
    }
    return op_operands;
  }();
  const auto& shardable_axes_sig = [&] {
    ShardableAxesSignature signature;
    int result_idx = GetOutputShardableAxesResultIdx(sink);
    signature.sole_output_sa = SoleOutputShardableAxes{
        .shardable_axes = value2shardable_axes.at(sink->result(result_idx)),
    };
    for (const auto& pair : input_op_operands) {
      const auto& [op, idx] = pair;
      pir::Value input = op->operand_source(idx);
      signature.input_shardable_axes[pair] = value2shardable_axes.at(input);
    }
    return signature;
  }();
  return shardable_axes_sig;
}
}  // namespace cinn::frontend::cluster_ops
