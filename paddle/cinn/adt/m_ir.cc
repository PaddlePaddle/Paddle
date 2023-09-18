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

namespace cinn::adt {

template <typename DoEachT>
void VisitEachOpStmt(const List<OpStmt>& op_stmts, const DoEachT& DoEach) {
  for (const auto& op_stmt_node : *op_stmts) {
    DoEach(op_stmt_node);
  }
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret);

void CollectTensorIndexIterators(const Undefined& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  LOG(FATAL) << "Not Implemented";
}

void CollectTensorIndexIterators(const Iterator& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  ret->insert(tensor_index_expr);
}

void CollectTensorIndexIterators(const List<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  for (const auto& value : *tensor_index_expr) {
    CollectTensorIndexIterators(value, ret);
  }
}

void CollectTensorIndexIterators(const IndexDot<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIteratorsValue(), ret);
}

void CollectTensorIndexIterators(const IndexUnDot<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIndexValue(), ret);
}

void CollectTensorIndexIterators(const ConstantAdd<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(const ConstantDiv<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(const ConstantMod<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const ListGetItem<Value, Constant>& tensor_index_expr,
    std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetList(), ret);
}

void CollectTensorIndexIterators(const PtrGetItem<Value>& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg1(), ret);
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret) {
  std::visit(
      [&](auto&& impl) { CollectTensorIndexIterators(std::move(impl), ret); },
      tensor_index_expr.variant());
}

std::unordered_set<Iterator> GetTensorIndexIterators(
    const TensorIndexExpr& tensor_index_expr) {
  std::unordered_set<Iterator> ret{};

  CollectTensorIndexIterators(tensor_index_expr, &ret);

  return ret;
}

LoopIterators GetLeftAlignedSdIterators(
    const std::unordered_set<Iterator>& tensor_index_loop_iters,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor*(const Iterator&)>&
        GetLoopDescriptor) {
  const auto& Used = [&](const Iterator& iter_var) {
    return tensor_index_loop_iters.count(iter_var) != 0;
  };

  const auto& IsIterVarLoopTypeSpatial = [&](const Iterator& iter_var) {
    return IsSpatial(GetLoopDescriptor(iter_var)->GetLoopType());
  };

  LoopIterators ret{loop_iters->begin(), loop_iters->end()};
  for (int i = ret->size() - 1; i >= 0; --i) {
    if (Used(ret->at(i)) || IsIterVarLoopTypeSpatial(ret->at(i))) {
      break;
    } else {
      ret->resize(i);
    }
  }
  return ret;
}

LoopIterators GetTensorLoopIterators(
    const Tensor& tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor*(const Iterator&)>&
        GetLoopDescriptor,
    const std::function<const TensorIndexExpr*(const Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& tensor_index_loop_iters =
      GetTensorIndexIterators(*TensorIndexExpr4Tensor(tensor));

  return GetLeftAlignedSdIterators(
      tensor_index_loop_iters, loop_iters, GetLoopDescriptor);
}

namespace {

std::unordered_map<Variable, const Value> MakeAnchorIndex2Ok(
    const Index& anchor_index) {
  return {{anchor_index, Ok{}}};
}

}  // namespace

bool LocalEquationsSolvable(const GraphView& graph_view,
                            const Index& anchor_index,
                            const FakeOpPlaceHolder& fake_op_placeholder) {
  const auto& init_var2value = MakeAnchorIndex2Ok(anchor_index);
  IndexExprInferContext ctx{init_var2value};
  bool has_no_conflict_value =
      TrySolveEquations(graph_view, anchor_index, &ctx).value();
  return has_no_conflict_value && ctx.HasValue(fake_op_placeholder);
}

std::vector<Index> GenerateWriteBroadcastTensorIndexs(
    config::NaiveOpEquationContext* ctx) {
  const auto& graph_view = Graph{ctx->equations()}.GetGraphView();
  std::vector<Index> ret{};
  const auto& fake_op_placeholder = ctx->fake_op_placeholder();
  ctx->VisitEachOutputTensorIndex([&](const auto& out_index) {
    if (!LocalEquationsSolvable(graph_view, out_index, fake_op_placeholder)) {
      ret.emplace_back(out_index);
    }
  });
  return ret;
}

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<config::NaiveOpEquationContext>(
        const OpStmt&)>;

void EraseWriteBroadcastOutMsgBox(
    const std::vector<Index>& truncated_output_tensor_indexes,
    config::NaiveOpEquationContext* ctx) {
  ctx->EraseOutMsgBoxIndexes(truncated_output_tensor_indexes);
}

void EraseWriteBroadcastOutMsgBoxes(
    const List<OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  VisitEachOpStmt(op_stmts, [&](const auto& op_stmt) {
    auto* ctx = EquationCtx4OpStmt(op_stmt).get();
    const auto& truncated_output_tensor_idxes =
        GenerateWriteBroadcastTensorIndexs(ctx);
    EraseWriteBroadcastOutMsgBox(truncated_output_tensor_idxes, ctx);
  });
}

namespace {

Tensor GetTensor(const OpStmt& op_stmt, const Undefined& undefined) {
  LOG(FATAL) << "position not found";
}

Tensor GetTensor(const OpStmt& op_stmt, const tIn<std::size_t>& pos) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  return in_args.value()->at(pos.value());
}

Tensor GetTensor(const OpStmt& op_stmt, const tOut<std::size_t>& pos) {
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  return out_args.value()->at(pos.value());
}

Tensor GetTensor(const config::NaiveOpEquationContext& ctx,
                 const OpStmt& op_stmt,
                 const Index& index) {
  const auto& op_arg_pos = ctx.GetOpArgPos(index);
  return std::visit([&](const auto& impl) { return GetTensor(op_stmt, impl); },
                    op_arg_pos.variant());
}

const auto& GetAnchorTensor(const AnchorGroup& anchor_group) {
  const auto& ctx = *anchor_group.EquationCtx4OpStmt(anchor_group.op_stmt);
  return GetTensor(ctx, anchor_group.op_stmt, anchor_group.anchor_index);
}

std::unordered_map<Index, LoopIterators> GenerateAnchorIndex2LoopIterators(
    const std::vector<AnchorGroup>& partitioned_anchor_groups,
    const std::function<const TensorIndexExpr*(const Tensor&)>&
        TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor*(const Iterator&)>&
        GetLoopDescriptor) {
  std::unordered_map<Index, LoopIterators> anchor_index2loop_iters{};
  for (const auto& anchor_group : partitioned_anchor_groups) {
    const auto& anchor_index = anchor_group.anchor_index;
    const auto& anchor_tensor = GetAnchorTensor(anchor_group);
    const auto& anchor_loop_iters = GetTensorLoopIterators(
        anchor_tensor, loop_iters, GetLoopDescriptor, TensorIndexExpr4Tensor);
    CHECK(anchor_index2loop_iters.emplace(anchor_index, anchor_loop_iters)
              .second);
  }
  return anchor_index2loop_iters;
}

struct IteratorsSlice {
  std::size_t start;
  std::size_t end;
};

std::set<std::size_t> GetLoopIteratorSizes(
    const std::unordered_map<Index, LoopIterators>& index2loop_iters) {
  const auto& VisitLoopIteratorsSize = [&](const auto& DoEach) {
    for (const auto& [_, loop_iters] : index2loop_iters) {
      DoEach(loop_iters->size());
    }
  };
  std::set<std::size_t> ret{};
  VisitLoopIteratorsSize([&](std::size_t size) { ret.emplace(size); });
  return ret;
}

std::vector<IteratorsSlice> GetLoopIteratorSlices(
    const std::unordered_map<Index, LoopIterators>& index2loop_iters) {
  std::set<std::size_t> loop_iter_sizes =
      GetLoopIteratorSizes(index2loop_iters);
  std::vector<IteratorsSlice> ret{};
  ret.reserve(index2loop_iters.size());
  std::size_t pos = 0;
  for (std::size_t size : loop_iter_sizes) {
    ret.push_back(IteratorsSlice{pos, size});
    pos = size;
  }
  return ret;
}

List<LoopIterators> MakeLoopItersList(
    const LoopIterators& loop_iters,
    const std::vector<IteratorsSlice>& slices) {
  List<LoopIterators> ret{};
  for (const auto& slice : slices) {
    const auto& begin = std::next(loop_iters->begin(), slice.start);
    const auto& end = std::next(loop_iters->begin(), slice.end);
    if (slice.start < loop_iters->size()) {
      ret->emplace_back(LoopIterators{begin, end});
    }
  }
  return ret;
}

}  // namespace

MapIrList ConvertAnchorGroups2MapIrList(
    const std::vector<AnchorGroup>& partitioned_anchor_groups,
    const std::function<const TensorIndexExpr*(const Tensor&)>&
        TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor*(const Iterator&)>&
        GetLoopDescriptor) {
  const auto& anchor_index2loop_iters =
      GenerateAnchorIndex2LoopIterators(partitioned_anchor_groups,
                                        TensorIndexExpr4Tensor,
                                        loop_iters,
                                        GetLoopDescriptor);
  std::vector<IteratorsSlice> loop_iter_slices =
      GetLoopIteratorSlices(anchor_index2loop_iters);
  MapIrList ret{};
  for (const auto& anchor_group : partitioned_anchor_groups) {
    const auto& anchor_index = anchor_group.anchor_index;
    const auto& anchor_loop_iters = anchor_index2loop_iters.at(anchor_index);
    const auto& loop_iters_list =
        MakeLoopItersList(anchor_loop_iters, loop_iter_slices);
    ret->emplace_back(MapIr{anchor_group.op_stmts, loop_iters_list});
  }
  return ret;
}

MapIrList GenerateMapIrListForLoopFuse(
    const List<OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor*(const Iterator&)>&
        GetLoopDescriptor,
    const std::function<const TensorIndexExpr*(const Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& EquationCtx4OpStmt =
      config::GenerateContext4LocalOpStmt(op_stmts);
  EraseWriteBroadcastOutMsgBoxes(op_stmts, EquationCtx4OpStmt);

  const auto& partitioned_anchor_groups =
      PartitionOpStmts(EquationCtx4OpStmt, op_stmts);

  return ConvertAnchorGroups2MapIrList(partitioned_anchor_groups,
                                       TensorIndexExpr4Tensor,
                                       loop_iters,
                                       GetLoopDescriptor);
}

}  // namespace cinn::adt
