// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/direction_equation_generator.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/equation_util.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/common/enforce.h"
namespace cinn::adt {

AnchorIndex PickThenEraseAnchorIndex(
    std::vector<AnchorIndex>* candidate_anchor_indexes) {
  AnchorIndex ret = candidate_anchor_indexes->back();
  candidate_anchor_indexes->pop_back();
  return ret;
}

std::vector<AnchorIndex> InitCandidateAnchorIndex(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts) {
  std::vector<AnchorIndex> ret{};
  for (const auto& op_stmt : *op_stmts) {
    const auto& equation_ctx = EquationCtx4OpStmt(op_stmt);
    equation_ctx->VisitEachTensorIndex(
        [&](const auto& tensor_index) { ret.emplace_back(tensor_index); });
  }
  return ret;
}

std::pair<std::optional<OpStmt>, List<OpStmt>> FindVisitedOpStmts(
    const AnchorIndex& anchor_index,
    const GraphView& equation_graph,
    const std::function<const OpStmt*(const FakeOpPlaceHolder&)>&
        OpStmt4OpPlaceHolder,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    std::unordered_set<Variable>* visited_variables,
    std::unordered_set<const void*>* visited_functions) {
  std::optional<OpStmt> opt_anchor_op_stmt{std::nullopt};
  List<OpStmt> visited_op_stmts{};
  const auto& TrySetAnchorOpStmt = [&](const auto& op_stmt) {
    const auto& op_arg_pos =
        EquationCtx4OpStmt(op_stmt)->GetOpArgPos(anchor_index);
    const bool valid = !op_arg_pos.template Has<Undefined>();
    if (valid) {
      PADDLE_ENFORCE_EQ(opt_anchor_op_stmt.has_value(),
                        false,
                        ::common::errors::InvalidArgument(
                            "The opt_anchor_op_stmt must not have a value."));
      opt_anchor_op_stmt = op_stmt;
    }
  };
  const auto& DoEach = [&](const Variable variable) {
    if (visited_variables != nullptr) {
      visited_variables->insert(variable);
    }
    if (variable.Has<FakeOpPlaceHolder>()) {
      const auto& fake_op_placeholder = variable.Get<FakeOpPlaceHolder>();
      const auto& op_stmt = *OpStmt4OpPlaceHolder(fake_op_placeholder);
      visited_op_stmts->emplace_back(op_stmt);
      TrySetAnchorOpStmt(op_stmt);
    }
  };
  const auto& DoEachFunction = [&](const Function* function) {
    if (visited_functions != nullptr) {
      visited_functions->insert(GetFunctionDataPtr(*function));
    }
  };
  std::array<AnchorIndex, 1> starts{anchor_index};

  equation_graph(starts.begin(), starts.end(), DoEach, DoEachFunction);
  return std::pair{opt_anchor_op_stmt, visited_op_stmts};
}

template <typename DoEachT>
void VisitEachEquation(const List<OpStmt>& op_stmts,
                       const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                       const DoEachT& DoEach) {
  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    ctx->VisitEachEquation(DoEach);
  }
}

GraphView MakeOpsGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts) {
  Equations equations{};

  VisitEachEquation(op_stmts, EquationCtx4OpStmt, [&](const auto& equation) {
    equations->emplace_back(equation);
  });

  return Graph<Variable, Equation>::New(equations)->GetGraphView();
}

template <typename DoEachT>
void VisitEachIndexAndAsOutput(const List<OpStmt>& op_stmts,
                               const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                               const DoEachT& DoEach) {
  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    ctx->VisitEachInputTensorIndex(
        [&](const auto& index) { DoEach(op_stmt, index, tOut<bool>{false}); });
    ctx->VisitEachOutputTensorIndex(
        [&](const auto& index) { DoEach(op_stmt, index, tOut<bool>{true}); });
  }
}

void MakeGetters4Indexes(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const std::shared_ptr<DirectionEquationGenerator>&
        direction_equation_generator,
    std::function<tOut<bool>(const Index&)>* AsOutput4Index,
    std::function<Index(const Index&)>* OutMsgIndex4InMsgIndex) {
  using Index2AsOutput = std::unordered_map<Index, tOut<bool>>;
  const auto& index2as_output = std::make_shared<Index2AsOutput>();

  const auto& UpdateCaches =
      [&](const auto& op_stmt, const auto& index, const auto& as_output) {
        PADDLE_ENFORCE_EQ(
            index2as_output->emplace(index, as_output).second,
            true,
            ::common::errors::InvalidArgument(
                "Failed to emplace the new element into index2as_output."));
      };

  VisitEachIndexAndAsOutput(op_stmts, EquationCtx4OpStmt, UpdateCaches);

  *AsOutput4Index = [index2as_output](const Index& index) {
    return index2as_output->at(index);
  };

  *OutMsgIndex4InMsgIndex =
      [direction_equation_generator](const Index& index) -> Index {
    const auto& out_msg_index =
        direction_equation_generator->OutMsgIndex4InMsgIndex(index);
    PADDLE_ENFORCE_EQ(out_msg_index.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The out_msg_index must not have a value."));
    return out_msg_index.value();
  };
}

std::unordered_map<Tensor, std::vector<Index>> GenerateSameTensor2Indexes(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  std::unordered_map<Tensor, std::vector<Index>> tensor2indexes;

  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = EquationCtx4OpStmt(op_stmt);
    const auto& [op, op_inputs, op_outputs] = op_stmt.tuple();
    for (std::size_t idx = 0; idx < op_inputs.value()->size(); ++idx) {
      tensor2indexes[op_inputs.value()->at(idx)].emplace_back(
          ctx->GetInIndex(idx));
    }
    for (std::size_t idx = 0; idx < op_outputs.value()->size(); ++idx) {
      tensor2indexes[op_outputs.value()->at(idx)].emplace_back(
          ctx->GetOutIndex(idx));
    }
  }

  return tensor2indexes;
}

template <typename DoEachT>
void VisitIndexesOfSameTensor(const List<OpStmt>& op_stmts,
                              const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                              const DoEachT& DoEach) {
  const auto& tensor2indexes =
      GenerateSameTensor2Indexes(op_stmts, EquationCtx4OpStmt);

  for (const auto& [tensor, indexes] : tensor2indexes) {
    DoEach(indexes);
  }
}

// DoEachT is like void(*)(Index producer_index, Index
// consumer_index)
template <typename AsOutput4IndexT, typename DoEachT>
void VisitProducerConsumerPair(const std::vector<Index>& tensor_indexes,
                               const AsOutput4IndexT& AsOutput4Index,
                               const DoEachT& DoEach) {
  PADDLE_ENFORCE_EQ(tensor_indexes.empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The tensor_indexes container must not be empty."));
  if (AsOutput4Index(tensor_indexes.at(0)).value()) {  // Write first
    auto producer = tensor_indexes.at(0);
    for (std::size_t idx = 1; idx < tensor_indexes.size(); ++idx) {
      DoEach(producer, tensor_indexes.at(idx));
      if (AsOutput4Index(tensor_indexes.at(idx)).value()) {
        producer = tensor_indexes.at(idx);
      }
    }
  } else {
    for (const auto& tensor_index : tensor_indexes) {  // Read first
      PADDLE_ENFORCE_EQ(
          AsOutput4Index(tensor_index).value(),
          false,
          ::common::errors::InvalidArgument(
              "The value returned by AsOutput4Index(tensor_index).value() must "
              "be false."));
    }
  }
}

// DoEachT is like void(*)(Index producer_index, Index
// consumer_index)
template <typename AsOutput4IndexT, typename DoEachT>
void VisitProducerConsumerTensorIndexPair(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const AsOutput4IndexT& AsOutput4Index,
    const DoEachT& DoEach) {
  VisitIndexesOfSameTensor(
      op_stmts, EquationCtx4OpStmt, [&](const auto& indexes) {
        VisitProducerConsumerPair(indexes, AsOutput4Index, DoEach);
      });
}

void CollectIdentity(const Index& in_tensor_index,
                     const Index& out_tensor_index,
                     Equations* equations) {
  IdentityConnect(out_tensor_index, in_tensor_index, equations);
}

GraphView MakeParametersGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts,
    const std::shared_ptr<DirectionEquationGenerator>&
        direction_equation_generator) {
  Equations equations{};

  std::function<tOut<bool>(const Index&)> AsOutput4Index{};
  std::function<Index(const Index&)> OutMsgIndex4InMsgIndex{};
  MakeGetters4Indexes(op_stmts,
                      EquationCtx4OpStmt,
                      direction_equation_generator,
                      &AsOutput4Index,
                      &OutMsgIndex4InMsgIndex);

  const auto& CollectEquation = [&](const auto& producer_index,
                                    const auto& consumer_index) {
    CollectIdentity(
        OutMsgIndex4InMsgIndex(producer_index), consumer_index, &equations);
    CollectIdentity(
        OutMsgIndex4InMsgIndex(consumer_index), producer_index, &equations);
  };
  VisitProducerConsumerTensorIndexPair(
      op_stmts, EquationCtx4OpStmt, AsOutput4Index, CollectEquation);

  return Graph<Variable, Equation>::New(equations)->GetGraphView();
}

GraphView MakeGlobalEquationGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts,
    const std::shared_ptr<DirectionEquationGenerator>&
        direction_equation_generator) {
  const auto& ops_graph_view =
      MakeOpsGraphViewForPartition(EquationCtx4OpStmt, op_stmts);

  const auto& direction_equation_view =
      Graph<Variable, Equation>::New(
          direction_equation_generator->GetDirectionEquations())
          ->GetGraphView();

  const auto& parameters_graph_view = MakeParametersGraphViewForPartition(
      EquationCtx4OpStmt, op_stmts, direction_equation_generator);

  return ops_graph_view.Merge(direction_equation_view)
      .Merge(parameters_graph_view);
}

template <typename DoEachT>
void VisitTensorIndex(const AnchorGroup& igroup_spec, const DoEachT& DoEach) {
  const auto& op_stmts = igroup_spec.op_stmts;
  const auto& EquationCtx4OpStmt = igroup_spec.EquationCtx4OpStmt;
  for (const auto& igroup_op_stmt : *op_stmts) {
    const auto& ctx = EquationCtx4OpStmt(igroup_op_stmt);
    ctx->VisitEachTensorIndex(DoEach);
  }
}

void CleanSmallAnchorGroups(
    const AnchorGroup& igroup_spec,
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec) {
  VisitTensorIndex(igroup_spec, [&](const auto& tensor_index) {
    anchor_index2igroup_spec->erase(tensor_index);
  });
}

void UpdateAnchorIndex2AnchorGroup(
    const AnchorGroup& igroup_spec,
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec) {
  CleanSmallAnchorGroups(igroup_spec, anchor_index2igroup_spec);

  PADDLE_ENFORCE_EQ(
      anchor_index2igroup_spec->emplace(igroup_spec.anchor_index, igroup_spec)
          .second,
      true,
      ::common::errors::InvalidArgument(
          "Failed to emplace the element into the "
          "map. The key might already exist."));
}

void EraseCandidateAnchorIndexes(
    const AnchorGroup& igroup_spec,
    std::vector<AnchorIndex>* candidate_anchor_indexes) {
  VisitTensorIndex(igroup_spec, [&](const auto& tensor_index) {
    auto iter = std::find(candidate_anchor_indexes->begin(),
                          candidate_anchor_indexes->end(),
                          tensor_index);
    while (iter != candidate_anchor_indexes->end()) {
      candidate_anchor_indexes->erase(iter);
      iter = std::find(candidate_anchor_indexes->begin(),
                       candidate_anchor_indexes->end(),
                       tensor_index);
    }
  });
}

std::unordered_map<AnchorIndex, AnchorGroup> PartitionOpStmtsIntoAnchorGroups(
    std::vector<AnchorIndex>* candidate_anchor_indexes,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts,
    const std::shared_ptr<DirectionEquationGenerator>&
        direction_equation_generator) {
  PADDLE_ENFORCE_EQ(op_stmts->empty(),
                    false,
                    ::common::errors::InvalidArgument(
                        "The op_stmts container must not be empty."));
  std::unordered_map<AnchorIndex, AnchorGroup> anchor_index2igroup_spec{};

  const auto& OpStmt4OpPlaceHolder =
      direction_equation_generator->MakeGetterOpStmt4OpPlaceHolder();

  const auto& equation_graph_view = MakeGlobalEquationGraphViewForPartition(
      EquationCtx4OpStmt, op_stmts, direction_equation_generator);

  Equations graph_equations{};
  {
    Variable start = *candidate_anchor_indexes->begin();
    equation_graph_view.BfsWalkFunction(
        start, [&](const Function* f) { graph_equations->emplace_back(*f); });
  }

  std::unordered_set<OpStmt> all_visited_op_stmts{};
  while (!candidate_anchor_indexes->empty()) {
    AnchorIndex anchor_index =
        PickThenEraseAnchorIndex(candidate_anchor_indexes);

    const auto& [opt_anchor_op_stmt, visited_op_stmts] =
        FindVisitedOpStmts(anchor_index,
                           equation_graph_view,
                           OpStmt4OpPlaceHolder,
                           EquationCtx4OpStmt,
                           /*visited_variables=*/nullptr,
                           /*visited_functions=*/nullptr);

    if (visited_op_stmts->empty()) {
      continue;
    }
    PADDLE_ENFORCE_EQ(opt_anchor_op_stmt.has_value(),
                      true,
                      ::common::errors::InvalidArgument(
                          "The opt_anchor_op_stmt must not have a value."));
    all_visited_op_stmts.insert(visited_op_stmts->begin(),
                                visited_op_stmts->end());

    AnchorGroup igroup_spec{anchor_index,
                            opt_anchor_op_stmt.value(),
                            visited_op_stmts,
                            EquationCtx4OpStmt};
    UpdateAnchorIndex2AnchorGroup(igroup_spec, &anchor_index2igroup_spec);

    EraseCandidateAnchorIndexes(igroup_spec, candidate_anchor_indexes);
  }

  PADDLE_ENFORCE_EQ(all_visited_op_stmts.size(),
                    op_stmts->size(),
                    ::common::errors::InvalidArgument(
                        "Some fake_op_placeholders are not visited."));
  return anchor_index2igroup_spec;
}

void AnchorGroup::PrintEquations() const {
  const auto& ctx = EquationCtx4OpStmt(op_stmt);
  VLOG(1) << "anchor_index: ";
  VLOG(1) << ToTxtString(anchor_index);
  VLOG(1) << "AnchorGroup.equations: ";
  // ctx->Print();
}

std::unordered_map<Variable, const Value> MakeAnchorIndex2Ok(
    const AnchorGroup& igroup_spec) {
  return {{igroup_spec.anchor_index, Ok{}}};
}

template <typename DoEachT>
tBreak<bool> AggregateAnchorGroupOpStmt(const AnchorGroup& igroup_spec,
                                        const DoEachT& DoEach) {
  for (const auto& op_stmt : *igroup_spec.op_stmts) {
    tBreak<bool> ret = DoEach(op_stmt);
    if (ret.value()) {
      return ret;
    }
  }
  return tBreak<bool>{false};
}

void CheckEquationSolvable(const AnchorGroup& igroup_spec,
                           const std::shared_ptr<DirectionEquationGenerator>&
                               direction_equation_generator) {
  const auto& equation_graph_view =
      MakeGlobalEquationGraphViewForPartition(igroup_spec.EquationCtx4OpStmt,
                                              igroup_spec.op_stmts,
                                              direction_equation_generator);

  const auto& init_var2value = MakeAnchorIndex2Ok(igroup_spec);
  IndexExprInferContext ctx{init_var2value};

  const auto& IsOpSolved = [&](const auto& op_stmt) {
    const auto& equation_ctx = *igroup_spec.EquationCtx4OpStmt(op_stmt);
    const auto& fake_op_placeholder = equation_ctx.fake_op_placeholder();
    return ctx.HasValue(fake_op_placeholder);
  };

  CheckEquationsSolvable(equation_graph_view, igroup_spec.anchor_index, &ctx);
  AggregateAnchorGroupOpStmt(igroup_spec, [&](const auto& op_stmt) {
    PADDLE_ENFORCE_EQ(IsOpSolved(op_stmt),
                      true,
                      ::common::errors::InvalidArgument(
                          "The operation statement must be solved."));
    return tBreak<bool>{false};
  });
}

std::function<std::size_t(const OpStmt&)> MakeGetterOrderValue4OpStmt(
    const List<OpStmt>& op_stmts) {
  using OpStmt2OrderValue = std::unordered_map<OpStmt, std::size_t>;
  const auto& op_stmt2order_value = std::make_shared<OpStmt2OrderValue>();
  for (std::size_t idx = 0; idx < op_stmts->size(); ++idx) {
    PADDLE_ENFORCE_EQ(
        op_stmt2order_value->emplace(op_stmts->at(idx), idx).second,
        true,
        ::common::errors::InvalidArgument(
            "Failed to emplace the element into op_stmt2order_value. The key "
            "might already exist."));
  }
  return [op_stmt2order_value](const auto& op_stmt) {
    return op_stmt2order_value->at(op_stmt);
  };
}

template <typename DoEachT>
void VisitEachAnchorGroup(
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec,
    const DoEachT& DoEach) {
  for (auto& [anchor_index, igroup_spec] : *anchor_index2igroup_spec) {
    DoEach(&igroup_spec);
  }
}

template <typename DoEachT>
void VisitEachAnchorGroup(const std::unordered_map<AnchorIndex, AnchorGroup>&
                              anchor_index2igroup_spec,
                          const DoEachT& DoEach) {
  for (const auto& [anchor_index, igroup_spec] : anchor_index2igroup_spec) {
    DoEach(igroup_spec);
  }
}

void SortAnchorGroupOpStmts(
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec,
    const std::function<std::size_t(const OpStmt&)>& OrderValue4OpStmt) {
  const auto& CompareOpStmt = [&](const auto& lhs, const auto& rhs) {
    return OrderValue4OpStmt(lhs) < OrderValue4OpStmt(rhs);
  };

  VisitEachAnchorGroup(anchor_index2igroup_spec, [&](auto* igroup_spec) {
    std::sort(igroup_spec->op_stmts->begin(),
              igroup_spec->op_stmts->end(),
              CompareOpStmt);
  });
}

std::vector<AnchorGroup> SortedAnchorGroups(
    const std::unordered_map<AnchorIndex, AnchorGroup>&
        anchor_index2igroup_spec,
    const std::function<std::size_t(const OpStmt&)>& OrderValue4OpStmt) {
  std::vector<AnchorGroup> ret{};

  VisitEachAnchorGroup(anchor_index2igroup_spec, [&](const auto& igroup_spec) {
    ret.emplace_back(igroup_spec);
  });

  const auto& OrderValue4AnchorGroup = [&](const AnchorGroup& igroup_spec) {
    return OrderValue4OpStmt(igroup_spec.op_stmts->back());
  };
  std::sort(ret.begin(), ret.end(), [&](const auto& lhs, const auto& rhs) {
    return OrderValue4AnchorGroup(lhs) < OrderValue4AnchorGroup(rhs);
  });

  return ret;
}

std::vector<AnchorGroup> SortedAnchorGroups(
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec,
    const List<OpStmt>& op_stmts) {
  const auto& OrderValue4OpStmt = MakeGetterOrderValue4OpStmt(op_stmts);

  SortAnchorGroupOpStmts(anchor_index2igroup_spec, OrderValue4OpStmt);

  return SortedAnchorGroups(*anchor_index2igroup_spec, OrderValue4OpStmt);
}

std::vector<AnchorGroup> PartitionOpStmts(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<OpStmt>& op_stmts,
    const std::shared_ptr<DirectionEquationGenerator>&
        direction_equation_generator) {
  std::vector<AnchorIndex> candidate_anchor_indexes =
      InitCandidateAnchorIndex(EquationCtx4OpStmt, op_stmts);

  std::unordered_map<AnchorIndex, AnchorGroup> anchor_index2igroup_spec =
      PartitionOpStmtsIntoAnchorGroups(&candidate_anchor_indexes,
                                       EquationCtx4OpStmt,
                                       op_stmts,
                                       direction_equation_generator);

  return SortedAnchorGroups(&anchor_index2igroup_spec, op_stmts);
}

}  // namespace cinn::adt
