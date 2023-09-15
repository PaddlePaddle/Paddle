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
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"
#include "paddle/cinn/adt/partition_op_stmts.h"

namespace cinn::adt::m_ir {

template <typename DoEachT>
void VisitEachOpStmt(const List<m_expr::OpStmt>& op_stmts,
                     const DoEachT& DoEach) {
  for (const auto& op_stmt_node : *op_stmts) {
    DoEach(op_stmt_node);
  }
}

void CollectTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret);

void CollectTensorIndexIterators(const Undefined& tensor_index_expr,
                                 std::unordered_set<equation::Iterator>* ret) {
  LOG(FATAL) << "Not Implemented";
}

void CollectTensorIndexIterators(const equation::Iterator& tensor_index_expr,
                                 std::unordered_set<equation::Iterator>* ret) {
  ret->insert(tensor_index_expr);
}

void CollectTensorIndexIterators(const List<equation::Value>& tensor_index_expr,
                                 std::unordered_set<equation::Iterator>* ret) {
  for (const auto& value : *tensor_index_expr) {
    CollectTensorIndexIterators(value, ret);
  }
}

void CollectTensorIndexIterators(
    const equation::IndexDot<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(
    const equation::IndexUnDot<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantAdd<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantDiv<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantMod<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ListGetItem<equation::Value, equation::Constant>&
        tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetList(), ret);
}

void CollectTensorIndexIterators(
    const equation::PtrGetItem<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg1(), ret);
}

void CollectTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr,
    std::unordered_set<equation::Iterator>* ret) {
  std::visit(
      [&](auto&& impl) { CollectTensorIndexIterators(std::move(impl), ret); },
      tensor_index_expr.variant());
}

std::unordered_set<equation::Iterator> GetTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr) {
  std::unordered_set<equation::Iterator> ret{};

  CollectTensorIndexIterators(tensor_index_expr, &ret);

  return ret;
}

LoopIterators GetLeftAlignedSdIterators(
    const std::unordered_set<equation::Iterator>& tensor_index_loop_iters,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::Iterator&)>&
        GetLoopDescriptor) {
  const auto& Used = [&](const equation::Iterator& iter_var) {
    return tensor_index_loop_iters.count(iter_var) != 0;
  };

  const auto& IsIterVarLoopTypeSpatial =
      [&](const equation::Iterator& iter_var) {
        return IsSpatial(GetLoopDescriptor(iter_var).GetLoopType());
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
    const m_expr::Tensor& tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::Iterator&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& tensor_index_loop_iters =
      GetTensorIndexIterators(TensorIndexExpr4Tensor(tensor));

  return GetLeftAlignedSdIterators(
      tensor_index_loop_iters, loop_iters, GetLoopDescriptor);
}

namespace {

std::unordered_map<equation::Variable, equation::Value> MakeAnchorIndex2Ok(
    const equation::Index& anchor_index) {
  return {{anchor_index, Ok{}}};
}

}  // namespace

bool LocalEquationsSolvable(
    const equation::GraphView& graph_view,
    const equation::Index& anchor_index,
    const equation::FakeOpPlaceHolder& fake_op_placeholder) {
  const auto& init_var2value = MakeAnchorIndex2Ok(anchor_index);
  equation::IndexExprInferContext ctx{init_var2value};
  // Note: namespace and function name conflict, maybe we do not need value
  // namespace
  bool has_no_conflict_value =
      equation::value::TrySolveEquations(graph_view, anchor_index, &ctx)
          .value();
  return has_no_conflict_value && ctx.HasValue(fake_op_placeholder);
}

std::vector<equation::Index> GenerateWriteBroadcastTensorIndexs(
    equation::config::NativeOpEquationContext* ctx) {
  const auto& graph_view = equation::Graph{ctx->equations()}.GetGraphView();
  std::vector<equation::Index> ret{};
  const auto& fake_op_placeholder = ctx->fake_op_placeholder();
  ctx->VisitEachOutputTensorIndex([&](const auto& out_index) {
    if (!LocalEquationsSolvable(graph_view, out_index, fake_op_placeholder)) {
      ret.emplace_back(out_index);
    }
  });
  return ret;
}

using EquationCtx4OpStmtT =
    std::function<std::shared_ptr<equation::config::NativeOpEquationContext>(
        const m_expr::OpStmt&)>;

void EraseWriteBroadcastOutMsgBox(
    const std::vector<equation::Index>& truncated_output_tensor_indexes,
    equation::config::NativeOpEquationContext* ctx) {
  ctx->EraseOutMsgBoxIndexes(truncated_output_tensor_indexes);
}

void EraseWriteBroadcastOutMsgBoxes(
    const List<m_expr::OpStmt>& op_stmts,
    const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
  VisitEachOpStmt(op_stmts, [&](const auto& op_stmt) {
    auto* ctx = EquationCtx4OpStmt(op_stmt).get();
    const auto& truncated_output_tensor_idxes =
        GenerateWriteBroadcastTensorIndexs(ctx);
    EraseWriteBroadcastOutMsgBox(truncated_output_tensor_idxes, ctx);
  });
}

namespace {

m_expr::Tensor GetTensor(const equation::config::NativeOpEquationContext& ctx,
                         const m_expr::OpStmt& op_stmt,
                         const equation::Index& index) {
  const auto& op_arg_pos = ctx.GetOpArgPos(index);
  const auto& [op, in_args, out_args] = op_stmt.tuple();
  return op_arg_pos >>
         match{
             [&](const Undefined&) -> m_expr::Tensor {
               LOG(FATAL) << "position not found";
             },
             // Note: structured binding cannot be captured in lambda, fix this
             // following below link:
             // https://stackoverflow.com/questions/46114214/lambda-implicit-capture-fails-with-variable-declared-from-structured-binding
             [&in_args = in_args](const tIn<std::size_t>& pos)
                 -> m_expr::Tensor { return in_args.value()->at(pos.value()); },
             [&out_args =
                  out_args](const tOut<std::size_t>& pos) -> m_expr::Tensor {
               return out_args.value()->at(pos.value());
             }};
}

const auto& GetAnchorTensor(const partition::AnchorGroup& anchor_group) {
  const auto& ctx = *anchor_group.EquationCtx4OpStmt(anchor_group.op_stmt);
  return GetTensor(ctx, anchor_group.op_stmt, anchor_group.anchor_index);
}

std::unordered_map<equation::Index, LoopIterators>
GenerateAnchorIndex2LoopIterators(
    const std::vector<partition::AnchorGroup>& partitioned_anchor_groups,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::Iterator&)>&
        GetLoopDescriptor) {
  std::unordered_map<equation::Index, LoopIterators> anchor_index2loop_iters{};
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
    const std::unordered_map<equation::Index, LoopIterators>&
        index2loop_iters) {
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
    const std::unordered_map<equation::Index, LoopIterators>&
        index2loop_iters) {
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
    const std::vector<partition::AnchorGroup>& partitioned_anchor_groups,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::Iterator&)>&
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
    const List<m_expr::OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::Iterator&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& EquationCtx4OpStmt =
      equation::config::GenerateContext4LocalOpStmt(op_stmts);
  EraseWriteBroadcastOutMsgBoxes(op_stmts, EquationCtx4OpStmt);

  const auto& partitioned_anchor_groups =
      partition::PartitionOpStmts(EquationCtx4OpStmt, op_stmts);

  return ConvertAnchorGroups2MapIrList(partitioned_anchor_groups,
                                       TensorIndexExpr4Tensor,
                                       loop_iters,
                                       GetLoopDescriptor);
}

}  // namespace cinn::adt::m_ir
