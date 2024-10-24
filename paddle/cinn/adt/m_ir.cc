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

#include <iterator>
#include <unordered_map>

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"
#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/write_broadcast_disabled_bidirection_equation_generator.h"

namespace cinn::adt {

template <typename DoEachT>
void VisitEachOpStmt(const List<OpStmt>& op_stmts, const DoEachT& DoEach) {
  for (const auto& op_stmt_node : *op_stmts) {
    DoEach(op_stmt_node);
  }
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret);

void CollectTensorIndexIteratorsImpl(const Undefined& tensor_index_expr,
                                     std::unordered_set<Iterator>* ret) {
  PADDLE_THROW(::common::errors::Unimplemented(
      "CollectTensorIndexIteratorsImpl is not implemented for Undefined tensor "
      "index expression. Please check your input."));
}

void CollectTensorIndexIteratorsImpl(const Ok& ok,
                                     std::unordered_set<Iterator>* ret) {
  PADDLE_THROW(::common::errors::Unimplemented(
      "CollectTensorIndexIteratorsImpl is not implemented for Ok state. Please "
      "ensure the function is correctly called."));
}

void CollectTensorIndexIteratorsImpl(const Iterator& iterator,
                                     std::unordered_set<Iterator>* ret) {
  ret->emplace(iterator);
}

void CollectTensorIndexIteratorsImpl(const DimExpr& constant,
                                     std::unordered_set<Iterator>* ret) {
  // Do nothing
}

void CollectTensorIndexIteratorsImpl(const List<Value>& tensor_index_expr,
                                     std::unordered_set<Iterator>* ret) {
  for (const auto& value : *tensor_index_expr) {
    CollectTensorIndexIterators(value, ret);
  }
}

void CollectTensorIndexIteratorsImpl(
    const IndexDotValue<Value, List<DimExpr>>& tensor_index_expr,
    std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIteratorsValue(), ret);
}

void CollectTensorIndexIteratorsImpl(
    const IndexUnDotValue<Value, List<DimExpr>>& tensor_index_expr,
    std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIndexValue(), ret);
}

void CollectTensorIndexIteratorsImpl(
    const ListGetItem<Value, DimExpr>& tensor_index_expr,
    std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetList(), ret);
}

void CollectTensorIndexIteratorsImpl(
    const BroadcastedIterator<Value, DimExpr>& broadcasted_iterator,
    std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(broadcasted_iterator.GetArg0(), ret);
}

void CollectTensorIndexIteratorsImpl(const PtrGetItem<Value>& tensor_index_expr,
                                     std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg1(), ret);
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  std::visit(
      [&](const auto& impl) { CollectTensorIndexIteratorsImpl(impl, ret); },
      tensor_index_expr.variant());
}

std::unordered_set<Iterator> GetTensorIndexIterators(
    const TensorIndexExpr& tensor_index_expr) {
  std::unordered_set<Iterator> ret{};

  CollectTensorIndexIterators(tensor_index_expr, &ret);

  return ret;
}

LoopIterators GetSortedSdIterators(
    const std::unordered_set<Iterator>& tensor_index_loop_iters,
    const LoopIterators& loop_iters) {
  LoopIterators ret{};
  for (const auto& loop_iter : *loop_iters) {
    if (tensor_index_loop_iters.count(loop_iter) > 0) {
      ret->emplace_back(loop_iter);
    }
  }
  return ret;
}

LoopIterators GetAnchorTensorLoopIterators(
    const Tensor& tensor,
    const LoopIterators& loop_iters,
    const std::function<TensorIndexExpr(const Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& tensor_index_loop_iters =
      GetTensorIndexIterators(TensorIndexExpr4Tensor(tensor));

  return GetSortedSdIterators(tensor_index_loop_iters, loop_iters);
}

namespace {

Tensor GetTensorImpl(const OpStmt& op_stmt, const Undefined& undefined) {
  PADDLE_THROW(::common::errors::Fatal("position not found"));
}

Tensor GetTensorImpl(const OpStmt& op_stmt, const tIn<std::size_t>& pos) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  return in_args.value()->at(pos.value());
}

Tensor GetTensorImpl(const OpStmt& op_stmt, const tOut<std::size_t>& pos) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  return out_args.value()->at(pos.value());
}

Tensor GetTensor(const config::NaiveOpEquationContext& ctx,
                 const OpStmt& op_stmt,
                 const Index& index) {
  const auto& op_arg_pos = ctx.GetOpArgPos(index);
  return std::visit(
      [&](const auto& impl) { return GetTensorImpl(op_stmt, impl); },
      op_arg_pos.variant());
}

Tensor GetAnchorTensor(const AnchorGroup& anchor_group) {
  const auto& ctx = *anchor_group.EquationCtx4OpStmt(anchor_group.op_stmt);
  return GetTensor(ctx, anchor_group.op_stmt, anchor_group.anchor_index);
}

std::unordered_map<Index, LoopIterators> GenerateAnchorIndex2LoopIterators(
    const std::vector<AnchorGroup>& partitioned_anchor_groups,
    const std::function<TensorIndexExpr(const Tensor&)>& TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters) {
  std::unordered_map<Index, LoopIterators> anchor_index2loop_iters{};
  for (const auto& anchor_group : partitioned_anchor_groups) {
    const auto& anchor_index = anchor_group.anchor_index;
    const auto& anchor_tensor = GetAnchorTensor(anchor_group);
    const auto& anchor_loop_iters = GetAnchorTensorLoopIterators(
        anchor_tensor, loop_iters, TensorIndexExpr4Tensor);
    PADDLE_ENFORCE_EQ(
        anchor_index2loop_iters.emplace(anchor_index, anchor_loop_iters).second,
        true,
        ::common::errors::AlreadyExists("The anchor index has already "
                                        "been associated with loop iters."));
  }
  return anchor_index2loop_iters;
}

}  // namespace

MapIrList ConvertAnchorGroups2MapIrList(
    const std::vector<AnchorGroup>& partitioned_anchor_groups,
    const std::function<TensorIndexExpr(const Tensor&)>& TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters) {
  const auto& anchor_index2loop_iters = GenerateAnchorIndex2LoopIterators(
      partitioned_anchor_groups, TensorIndexExpr4Tensor, loop_iters);
  MapIrList ret{};
  for (const auto& anchor_group : partitioned_anchor_groups) {
    const auto& anchor_index = anchor_group.anchor_index;
    const auto& anchor_loop_iters = anchor_index2loop_iters.at(anchor_index);
    ret->emplace_back(MapIr{anchor_group.op_stmts, anchor_loop_iters});
  }
  return ret;
}

MapIrList GenerateMapIrListForLoopFuse(
    const List<OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<TensorIndexExpr(const Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& EquationCtx4OpStmt =
      config::GenerateContext4LocalOpStmt(op_stmts);
  auto direction_equation_generator =
      std::make_shared<WriteBroadcastDisabledBidirectionEquationGenerator>(
          op_stmts, EquationCtx4OpStmt);
  const auto& partitioned_anchor_groups = PartitionOpStmts(
      EquationCtx4OpStmt, op_stmts, direction_equation_generator);
  return ConvertAnchorGroups2MapIrList(
      partitioned_anchor_groups, TensorIndexExpr4Tensor, loop_iters);
}

}  // namespace cinn::adt
