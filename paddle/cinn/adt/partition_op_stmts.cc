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

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/equation_util.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"

namespace cinn::adt::partition {

using TensorIndex = equation::Variable;

AnchorIndex PickThenEraseAnchorIndex(
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes) {
  // Heuristic optimization will be added later
  // such as choosing the one with the biggest rank number as the anchor tensor
  // first
  const auto& ret = *candidate_anchor_indexes->begin();
  candidate_anchor_indexes->erase(candidate_anchor_indexes->begin());
  return ret;
}

std::unordered_set<AnchorIndex> InitCandidateAnchorIndex(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::unordered_set<AnchorIndex> ret{};
  for (const auto& op_stmt : *op_stmts) {
    const std::shared_ptr<equation::config::NativeOpEquationContext>
        equation_ctx = EquationCtx4OpStmt(op_stmt);
    equation_ctx->VisitEachTensorIndex(
        [&](const auto& tensor_index) { ret.emplace(tensor_index); });
  }
  return ret;
}

std::function<const m_expr::OpStmt&(const equation::FakeOpPlaceHolder&)>
MakeGetterOpStmt4OpPlaceHolder(const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                               const List<m_expr::OpStmt>& op_stmts) {
  using FakeOpPlaceHolder2OpStmt =
      std::unordered_map<const equation::FakeOpPlaceHolder, m_expr::OpStmt>;
  const auto& fake_op_placeholder2op_stmt =
      std::make_shared<FakeOpPlaceHolder2OpStmt>();

  for (const auto& op_stmt : *op_stmts) {
    const std::shared_ptr<equation::config::NativeOpEquationContext> ctx =
        EquationCtx4OpStmt(op_stmt);
    CHECK(fake_op_placeholder2op_stmt
              ->emplace(ctx->fake_op_placeholder(), op_stmt)
              .second);
  }

  return [fake_op_placeholder2op_stmt](
             const equation::FakeOpPlaceHolder& fake_op_placeholder) {
    return fake_op_placeholder2op_stmt->at(fake_op_placeholder);
  };
}

List<m_expr::OpStmt> FindVisitedOpStmts(
    const AnchorIndex& anchor_index,
    const equation::GraphView& equation_graph,
    const std::function<const m_expr::OpStmt&(
        const equation::FakeOpPlaceHolder&)>& OpStmt4OpPlaceHolder) {
  List<m_expr::OpStmt> ret{};
  equation_graph.WalkVariable(
      anchor_index, [&](const equation::Variable variable) {
        if (variable.Has<equation::FakeOpPlaceHolder>()) {
          ret->emplace_back(OpStmt4OpPlaceHolder(
              variable.Get<equation::FakeOpPlaceHolder>()));
        }
      });
  return ret;
}

template <typename DoEachT>
void VisitEachEquation(const List<m_expr::OpStmt>& op_stmts,
                       const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                       const DoEachT& DoEach) {
  for (const auto& op_stmt : *op_stmts) {
    const auto* ctx = EquationCtx4OpStmt(op_stmt);
    ctx->VisitEachEquation(DoEach);
  }
}

equation::GraphView MakeOpsGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  equation::Equations equations{};

  VisitEachEquation(op_stmts, EquationCtx4OpStmt, [&](const auto& equation) {
    equations->emplace_back(equation);
  });

  return std::make_shared<equation::Graph>(equations)->GetGraphView();
}

template <typename DoEachT>
void VisitEachIndexAndAsOutput(const List<m_expr::OpStmt>& op_stmts,
                               const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                               const DoEachT& DoEach) {
  for (const auto& op_stmt : op_stmts) {
    const auto* ctx = EquationCtx4OpStmt(op_stmt);
    ctx->VisitEachInputTensorIndex([&](const auto& index) {
      DoEach(op_stmt, index, tAsOutput<bool>{false});
    });
    ctx->VisitEachOutputTensorIndex([&](const auto& index) {
      DoEach(op_stmt, index, tAsOutput<bool>{true});
    });
  }
}

void MakeGetters4Indexes(
    const List<m_expr::OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    std::function<tAsOutput<bool>(const equation::Index&)>* AsOutput4Index,
    std::function<equation::Index(const equation::Index&)>*
        OutMsgBoxIndex4InMsgBoxIndex) {
  using Index2AsOutput =
      std::unordered_map<const equation::Index, tAsOutput<bool>>;
  const auto& index2as_output = std::make_shared<Index2AsOutput>();

  using Index2OwnerOpStmt =
      std::unordered_map<const equation::Index, m_expr::OpStmt>;
  const auto& index2owner_op_stmt = std::make_shared<Index2OwnerOpStmt>();

  const auto& UpdateCaches =
      [&](const auto& op_stmt, const auto& index, const auto& as_output) {
        CHECK(index2as_output->emplace(index, as_output).second);
        CHECK(index2owner_op_stmt->emplace(index, op_stmt).second);
      };

  VisitEachIndexAndAsOutput(op_stmts, EquationCtx4OpStmt, UpdateCaches);

  *AsOutput4Index = [index2as_output](const equation::Index& index) {
    return index2as_output->at(index);
  };

  *OutMsgBoxIndex4InMsgBoxIndex =
      [index2owner_op_stmt,
       EquationCtx4OpStmt](const equation::Index& index) -> equation::Index {
    const auto& op_stmt = index2owner_op_stmt->at(index);
    const std::shared_ptr<equation::config::NativeOpEquationContext> ctx =
        EquationCtx4OpStmt(op_stmt);
    const auto& out_msg_box_index = ctx->OutMsgBoxIndex4InMsgBoxIndex(index);
    CHECK(out_msg_box_index.has_value());
    return out_msg_box_index.value();
  };
}

std::unordered_map<m_expr::Tensor, std::vector<equation::Index>>
GenerateSameTensor2Indexes(const List<m_expr::OpStmt>& op_stmts,
                           const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  std::unordered_map<m_expr::Tensor, std::vector<equation::Index>>
      tensor2indexes;

  for (const auto& op_stmt : *op_stmts) {
    const std::shared_ptr<equation::config::NativeOpEquationContext> ctx =
        EquationCtx4OpStmt(op_stmt);
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
void VisitIndexesOfSameTensor(const List<m_expr::OpStmt>& op_stmts,
                              const EquationCtx4OpStmtT& EquationCtx4OpStmt,
                              const DoEachT& DoEach) {
  const auto& tensor2indexes =
      GenerateSameTensor2Indexes(op_stmts, EquationCtx4OpStmt);

  for (const auto& [tensor, indexes] : tensor2indexes) {
    DoEach(indexes);
  }
}

// DoEachT is like void(*)(equation::Index producer_index, equation::Index
// consumer_index)
template <typename AsOutput4IndexT, typename DoEachT>
void VisitProducerConsumerPair(
    const std::vector<equation::Index>& tensor_indexes,
    const AsOutput4IndexT& AsOutput4Index,
    const DoEachT& DoEach) {
  CHECK(!tensor_indexes.empty());
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
      CHECK(!AsOutput4Index(tensor_index).value());
    }
  }
}

// DoEachT is like void(*)(equation::Index producer_index, equation::Index
// consumer_index)
template <typename AsOutput4IndexT, typename DoEachT>
void VisitProducerConsumerTensorIndexPair(
    const List<m_expr::OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const AsOutput4IndexT& AsOutput4Index,
    const DoEachT& DoEach) {
  VisitIndexesOfSameTensor(
      op_stmts, EquationCtx4OpStmt, [&](const auto& indexes) {
        VisitProducerConsumerPair(indexes, AsOutput4Index, DoEach);
      });
}

void CollectIdentity(const equation::Index& in_tensor_index,
                     const equation::Index& out_tensor_index,
                     equation::Equations* equations) {
  equation::util::IdentityConnect(out_tensor_index, in_tensor_index, equations);
}

equation::GraphView MakeParametersGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  equation::Equations equations{};

  std::function<tAsOutput<bool>(const equation::Index&)> AsOutput4Index{};
  std::function<equation::Index(const equation::Index&)>
      OutMsgBoxIndex4InMsgBoxIndex{};
  MakeGetters4Indexes(op_stmts,
                      EquationCtx4OpStmt,
                      &AsOutput4Index,
                      &OutMsgBoxIndex4InMsgBoxIndex);

  const auto& CollectEquation = [&](const auto& producer_index,
                                    const auto& consumer_index) {
    CollectIdentity(OutMsgBoxIndex4InMsgBoxIndex(producer_index),
                    consumer_index,
                    &equations);
    CollectIdentity(OutMsgBoxIndex4InMsgBoxIndex(consumer_index),
                    producer_index,
                    &equations);
  };
  VisitProducerConsumerTensorIndexPair(
      op_stmts, EquationCtx4OpStmt, AsOutput4Index, CollectEquation);

  return std::make_shared<equation::Graph>(equations)->GetGraphView();
}

equation::GraphView MakeGlobalEquationGraphViewForPartition(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  const auto& ops_graph_view =
      MakeOpsGraphViewForPartition(EquationCtx4OpStmt, op_stmts);

  const auto& parameters_graph_view =
      MakeParametersGraphViewForPartition(EquationCtx4OpStmt, op_stmts);

  return ops_graph_view.Merge(parameters_graph_view);
}

template <typename DoEachT>
void VisitTensorIndex(const AnchorGroup& igroup_spec, const DoEachT& DoEach) {
  const auto& op_stmts = igroup_spec.op_stmts;
  const auto& EquationCtx4OpStmt = igroup_spec.EquationCtx4OpStmt;
  for (const auto& igroup_op_stmt : *op_stmts) {
    const auto* ctx = EquationCtx4OpStmt(igroup_op_stmt);
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

void UpdataAnchorIndex2AnchorGroup(
    const AnchorGroup& igroup_spec,
    std::unordered_map<AnchorIndex, AnchorGroup>* anchor_index2igroup_spec) {
  CleanSmallAnchorGroups(igroup_spec, anchor_index2igroup_spec);

  CHECK(anchor_index2igroup_spec->emplace(igroup_spec.anchor_index, igroup_spec)
            .second);
}

void EraseCandidateAnchorIndexes(
    const AnchorGroup& igroup_spec,
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes) {
  VisitTensorIndex(igroup_spec, [&](const auto& tensor_index) {
    candidate_anchor_indexes->erase(tensor_index);
  });
}

std::unordered_map<AnchorIndex, AnchorGroup> PartitionOpStmtsIntoAnchorGroups(
    std::unordered_set<AnchorIndex>* candidate_anchor_indexes,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::unordered_map<AnchorIndex, AnchorGroup> anchor_index2igroup_spec{};

  const auto& OpStmt4OpPlaceHolder =
      MakeGetterOpStmt4OpPlaceHolder(EquationCtx4OpStmt, op_stmts);

  const auto& equation_graph_view =
      MakeGlobalEquationGraphViewForPartition(EquationCtx4OpStmt, op_stmts);

  while (!candidate_anchor_indexes->empty()) {
    AnchorIndex anchor_tensor =
        PickThenEraseAnchorIndex(candidate_anchor_indexes);

    const auto& visited_op_stmts = FindVisitedOpStmts(
        anchor_tensor, equation_graph_view, OpStmt4OpPlaceHolder);
    CHECK(!visited_op_stmts->empty());

    AnchorGroup igroup_spec{
        anchor_tensor, visited_op_stmts, EquationCtx4OpStmt};
    UpdataAnchorIndex2AnchorGroup(igroup_spec, &anchor_index2igroup_spec);
    EraseCandidateAnchorIndexes(igroup_spec, candidate_anchor_indexes);
  }

  return anchor_index2igroup_spec;
}

std::unordered_map<const equation::Variable, equation::Value>
MakeAnchorIndex2Ok(const partition::AnchorGroup& igroup_spec) {
  return {{igroup_spec.anchor_index, Ok{}}};
}

bool IsEquationSolvable(const partition::AnchorGroup& igroup_spec) {
  const auto& equation_graph_view = MakeGlobalEquationGraphViewForPartition(
      igroup_spec.EquationCtx4OpStmt, igroup_spec.op_stmts);

  const auto& init_var2value = MakeAnchorIndex2Ok(igroup_spec);
  auto ctx = std::make_shared<equation::IndexExprInferContext>(init_var2value);

  return equation::value::IsEquationsSolvable(
      equation_graph_view, igroup_spec.anchor_index, ctx.get());
}

std::function<std::size_t(const m_expr::OpStmt&)> MakeGetterOrderValue4OpStmt(
    const List<m_expr::OpStmt>& op_stmts) {
  using OpStmt2OrderValue =
      std::unordered_map<const m_expr::OpStmt, std::size_t>;
  const auto& op_stmt2order_value = std::make_shared<OpStmt2OrderValue>();
  for (std::size_t idx = 0; idx < op_stmts->size(); ++idx) {
    CHECK(op_stmt2order_value->emplace(op_stmts->at(idx), idx).second);
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
    const std::function<std::size_t(const m_expr::OpStmt&)>&
        OrderValue4OpStmt) {
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
    const std::function<std::size_t(const m_expr::OpStmt&)>&
        OrderValue4OpStmt) {
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
    const List<m_expr::OpStmt>& op_stmts) {
  const auto& OrderValue4OpStmt = MakeGetterOrderValue4OpStmt(op_stmts);

  SortAnchorGroupOpStmts(anchor_index2igroup_spec, OrderValue4OpStmt);

  return SortedAnchorGroups(*anchor_index2igroup_spec, OrderValue4OpStmt);
}

std::vector<AnchorGroup> PartitionOpStmts(
    const EquationCtx4OpStmtT& EquationCtx4OpStmt,
    const List<m_expr::OpStmt>& op_stmts) {
  std::unordered_set<AnchorIndex> candidate_anchor_indexes =
      InitCandidateAnchorIndex(EquationCtx4OpStmt, op_stmts);

  std::unordered_map<AnchorIndex, AnchorGroup> anchor_index2igroup_spec =
      PartitionOpStmtsIntoAnchorGroups(
          &candidate_anchor_indexes, EquationCtx4OpStmt, op_stmts);

  return SortedAnchorGroups(&anchor_index2igroup_spec, op_stmts);
}

}  // namespace cinn::adt::partition
