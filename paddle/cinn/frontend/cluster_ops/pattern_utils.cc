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

#include "paddle/cinn/frontend/cluster_ops/pattern_utils.h"

namespace cinn::frontend::cluster_ops {

bool IsISPattern(const StmtPattern& pattern) {
  return std::holds_alternative<IS>(pattern);
}

bool IsPSPattern(const StmtPattern& pattern) {
  return std::holds_alternative<PS>(pattern);
}

bool IsRPattern(const StmtPattern& pattern) {
  return std::holds_alternative<R>(pattern);
}

pir::Value GetStmtBigestShapeValueImpl(const IS& injective_source) {
  const auto* sink_op = injective_source.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValueImpl(const R& reduce_pattern) {
  const auto* sink_op = reduce_pattern.reduce_op_pattern.reduce_op;
  CHECK_EQ(sink_op->num_operands(), 1);
  return sink_op->operand_source(0);
}

pir::Value GetStmtBigestShapeValueImpl(const PS& partial_shardable) {
  const auto* sink_op = partial_shardable.sole_sink;
  const int result_idx = GetOutputShardableAxesResultIdx(sink_op);
  return sink_op->result(result_idx);
}

pir::Value GetStmtBigestShapeValue(const StmtPattern& stmt) {
  return std::visit(
      [&](const auto& impl) { return GetStmtBigestShapeValueImpl(impl); },
      stmt);
}

const pir::Operation* GetStmtSoleSinkImpl(const IS& injective_source) {
  return injective_source.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const PS& partial_shardable) {
  return partial_shardable.sole_sink;
}

const pir::Operation* GetStmtSoleSinkImpl(const R& reduce) {
  return reduce.reduce_op_pattern.reduce_op;
}

const pir::Operation* GetStmtSoleSinkOp(const StmtPattern& stmt) {
  return std::visit([](const auto& impl) { return GetStmtSoleSinkImpl(impl); },
                    stmt);
}

void SortStmtPtrs(
    std::vector<const StmtPattern*>* stmt_ptrs,
    const std::function<size_t(const pir::Operation*)>& OrderValue4Op) {
  auto GetOrderValue4Stmt = [&](const StmtPattern* stmt) {
    const auto* sink_op = GetStmtSoleSinkOp(*stmt);
    return OrderValue4Op(sink_op);
  };
  const auto Cmp = [&](const auto* lhs, const auto* rhs) {
    const auto& lhs_order = GetOrderValue4Stmt(lhs);
    const auto& rhs_order = GetOrderValue4Stmt(rhs);
    return lhs_order < rhs_order;
  };
  std::sort(stmt_ptrs->begin(), stmt_ptrs->end(), Cmp);
}
common::TopoWalker<const StmtPattern*> MakeTopoWalker(
    const OpTopo& op_topo, const std::vector<StmtPattern>& stmt_patterns) {
  using StmtPtrs = std::vector<const StmtPattern*>;
  using Op2OwnerStmtPtrs = std::unordered_map<const pir::Operation*, StmtPtrs>;
  auto op2owner_stmt_ptr = std::make_shared<Op2OwnerStmtPtrs>();
  for (const auto& stmt : stmt_patterns) {
    VisitStmtOp(stmt, [&](const pir::Operation* op) {
      (*op2owner_stmt_ptr)[op].push_back(&stmt);
    });
  }
  using NodeVisitor = std::function<void(const StmtPattern*)>;
  auto VisitInput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    VisitStmtOp(*stmt, [&](const auto* op) {
      op_topo.VisitInputOp(op, [&](const auto* input_op) {
        const auto& owners_iter = op2owner_stmt_ptr->find(input_op);
        if (owners_iter == op2owner_stmt_ptr->end()) return;
        if (owners_iter->second.size() != 1) return;
        const auto* owner_stmt = *owners_iter->second.begin();
        if (owner_stmt == stmt) return;
        DoEach(owner_stmt);
      });
    });
  };
  auto VisitOutput = [=](const StmtPattern* stmt, const NodeVisitor& DoEach) {
    const auto* sink = GetStmtSoleSinkOp(*stmt);
    op_topo.VisitOutputOp(sink, [&](const pir::Operation* op) {
      const auto& owners_iter = op2owner_stmt_ptr->find(op);
      if (owners_iter == op2owner_stmt_ptr->end()) return;
      for (const StmtPattern* stmt : owners_iter->second) {
        DoEach(stmt);
      }
    });
  };
  const auto& TryPushBack = [](const auto* stmt, auto* stmts) {
    if (std::find(stmts->begin(), stmts->end(), stmt) == stmts->end()) {
      stmts->push_back(stmt);
    }
  };
  using EdgeCache =
      std::unordered_map<const StmtPattern*, std::vector<const StmtPattern*>>;
  auto stmt2inputs = std::make_shared<EdgeCache>();
  auto stmt2outputs = std::make_shared<EdgeCache>();
  for (const auto& stmt : stmt_patterns) {
    (void)(*stmt2inputs)[&stmt];
    VisitInput(&stmt, [&](const auto* input) {
      TryPushBack(input, &(*stmt2inputs)[&stmt]);
    });
    (void)(*stmt2outputs)[&stmt];
    VisitOutput(&stmt, [&](const auto* output) {
      TryPushBack(output, &(*stmt2outputs)[&stmt]);
    });
  }

  auto VisitCachedInput = [stmt2inputs](const auto* stmt,
                                        const NodeVisitor& DoEach) {
    const auto& map = (*stmt2inputs);
    const auto& iter = map.find(stmt);
    if (iter == map.end()) return;
    for (const auto* input : iter->second) {
      DoEach(input);
    }
  };
  auto VisitCachedOutput = [stmt2outputs](const auto* stmt,
                                          const NodeVisitor& DoEach) {
    const auto& map = (*stmt2outputs);
    const auto& iter = map.find(stmt);
    if (iter == map.end()) return;
    for (const auto* output : iter->second) {
      DoEach(output);
    }
  };
  return common::TopoWalker<const StmtPattern*>(VisitCachedInput,
                                                VisitCachedOutput);
}

std::function<bool(const pir::Operation*)> MakePredicatorIsInjectiveSource(
    const OpTopo& op_topo) {
  const auto& IsSource = [&](const pir::Operation* op) {
    std::size_t num_inputs = 0;
    op_topo.VisitInputOp(op,
                         [&](const pir::Operation* input) { ++num_inputs; });
    return num_inputs == 0;
  };

  const auto starts = [&] {
    std::list<const pir::Operation*> starts;
    for (const auto* op : *op_topo.ops) {
      if (IsSource(op)) {
        starts.push_back(op);
      } else {
        // do nothing.
      }
    }
    return starts;
  }();

  std::unordered_map<const pir::Operation*, bool> op_2_is_injective_source;

  auto IsInputsAllInjectiveSource = [&](const pir::Operation* op) {
    bool is_inputs_all_injective_source = true;
    op_topo.VisitInputOp(op, [&](const pir::Operation* input) {
      is_inputs_all_injective_source = (is_inputs_all_injective_source &&
                                        op_2_is_injective_source.at(input));
    });
    return is_inputs_all_injective_source;
  };
  const auto VisitInput = [&](const pir::Operation* op,
                              const OpVisitor& DoEach) {
    op_topo.VisitInputOp(op, DoEach);
  };
  const auto VisitOutput = [&](const pir::Operation* op,
                               const OpVisitor& DoEach) {
    op_topo.VisitOutputOp(op, DoEach);
  };
  common::TopoWalker<const pir::Operation*> walker{VisitInput, VisitOutput};
  walker(starts.begin(), starts.end(), [&](const pir::Operation* op) {
    op_2_is_injective_source[op] =
        (IsGeneralInjective(op) && IsInputsAllInjectiveSource(op));
  });
  return [map = std::move(op_2_is_injective_source)](const pir::Operation* op) {
    const auto& iter = map.find(op);
    CHECK(iter != map.end());
    return iter->second;
  };
}

std::vector<const pir::Operation*> GetStmtContainedOpsImpl(
    std::monostate nothing) {
  return {};
}

std::vector<const pir::Operation*> GetStmtContainedOpsImpl(
    const IS& injective_source) {
  return injective_source.ops;
}

std::vector<const pir::Operation*> GetStmtContainedOpsImpl(
    const PS& partial_shardable) {
  return partial_shardable.ops;
}

std::vector<const pir::Operation*> GetStmtContainedOpsImpl(const R& reduce) {
  const auto get_input_ops = [](std::variant<std::monostate, IS, PS> input) {
    return std::visit(
        [](const auto& impl) -> std::vector<const pir::Operation*> {
          return GetStmtContainedOpsImpl(impl);
        },
        input);
  };
  std::vector<const pir::Operation*> result = get_input_ops(reduce.input);
  result.emplace_back(reduce.reduce_op_pattern.reduce_op);
  return result;
}

std::vector<const pir::Operation*> GetStmtContainedOps(
    const StmtPattern& stmt) {
  return std::visit(
      [](const auto& impl) { return GetStmtContainedOpsImpl(impl); }, stmt);
}

std::string StmtPatternDebugStr(const StmtPattern& stmt) {
  std::stringstream ss;
  const auto& all_ops = GetStmtContainedOps(stmt);
  ss << "StmtPattern, size " << all_ops.size() << " :\n";
  ss << OpsDebugStr(all_ops);
  return ss.str();
}

std::string LoopAlignableStmtPatternVec::DebugStr() const {
  std::stringstream ss;
  ss << "Alignable Stmts, size " << stmts.size() << " :\n";
  for (const auto& stmt : stmts) {
    ss << StmtPatternDebugStr(stmt);
  }
  return ss.str();
}

std::string ClusteringResult::DebugStr() const {
  std::stringstream ss;
  ss << "Cluster Result:\n";
  for (const auto& alignable_stmt : loop_alignable_list) {
    ss << alignable_stmt.DebugStr();
  }
  return ss.str();
}

}  // namespace cinn::frontend::cluster_ops
