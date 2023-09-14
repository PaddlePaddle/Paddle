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
void MapIr::VisitEachTensor(const DoEachT& DoEach) const {
  ForEachTensor([&](const auto& tensor, const auto& as_output) {
    DoEach(tensor, as_output);
    return tBreak<bool>{false};
  });
}

template <typename DoEachT>
tBreak<bool> MapIr::ForEachTensor(const DoEachT& DoEach) const {
  for (const auto& op_node : ops_) {
    const auto& [op, inputs, outputs] = op_node.tuple();
    for (const auto& input : inputs.value()) {
      tBreak<bool> tag = DoEach(input, false);
      if (tag.value()) {
        return tag;
      }
    }
    for (const auto& output : outputs.value()) {
      tBreak<bool> tag = DoEach(output, true);
      if (tag.value()) {
        return tag;
      }
    }
  }
  return tBreak<bool>{false};
}

std::unordered_map<m_expr::Tensor, tOut<bool>> MapIr::GetTensor2AsOutput()
    const {
  std::unordered_map<m_expr::Tensor, tOut<bool>> ret{};

  VisitEachTensor([&](const m_expr::Tensor& tensor, tOut<bool> as_output) {
    ret[tensor] = ret[tensor].value() || as_output.value();
  });

  return ret;
}

template <typename DoEachT>
tBreak<bool> MapIr::AggregateTensorPair(const MapIr& that,
                                        const DoEachT& DoEach) const {
  auto that_tensor2as_output = that.GetTensor2AsOutput();

  return ForEachTensor([&](const auto& this_tensor,
                           const auto& this_as_output) {
    const auto& pair = that_tensor2as_output.find(this_tensor);
    if (pair != that_tensor2as_output.end()) {
      const auto& [that_tensor, that_as_output] = *pair;
      tBreak<bool> tag = DoEach(this_tensor, this_as_output, that_as_output);
      if (tag.value()) {
        return tag;
      }
    }
    return tBreak<bool>{false};
  });
}

bool MapIr::IsMergableTo(
    const MapIr& that,
    const std::function<const LoopIterators&(const m_expr::Tensor&)>&
        SdIterators4Tensor) const {
  // TODO(Hongyu Jia): Refact to support complicated cases
  if (that.loop_iters()->size() < this->loop_iters()->size()) {
    return false;
  }

  const auto& CheckBroadcast = [&](const auto& tensor) {
    return SdIterators4Tensor(tensor).size() < that.loop_iters().size();
  };

  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };
  bool mergable = true;
  const auto& UpdataMergable = [&](const auto& tensor,
                                   tOut<bool> this_as_output,
                                   tOut<bool> that_as_output) {
    if (CheckBroadcast(tensor) && CheckWrite(this_as_output, that_as_output)) {
      mergable = false;
      return tBreak<bool>{true};
    } else {
      return tBreak<bool>{false};
    }
  };
  AggregateTensorPair(that, UpdataMergable);
  return mergable;
}

bool MapIr::HasReadWriteDependence(const MapIr& that) const {
  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };

  bool has_read_write_dependence = false;

  AggregateTensorPair(that,
                      [&](const auto& tensor,
                          tOut<bool> this_as_output,
                          tOut<bool> that_as_output) {
                        if (CheckWrite(this_as_output, that_as_output)) {
                          has_read_write_dependence = true;
                          return tBreak<bool>{true};
                        } else {
                          return tBreak<bool>{false};
                        }
                      });

  return has_read_write_dependence;
}

void MapIr::MergeThisToThat(const MapIr& that) {
  CHECK_GE(that.loop_iters()->size(), this->loop_iters()->size());
  that.op_stmts_.splice(that.op_stmts_.begin(), std::move(this->op_stmts_));
}

template <typename DoEachT>
void VisitEachOpStmt(const List<m_expr::OpStmt>& op_stmts,
                     const DoEachT& DoEach) {
  for (const auto& op_stmt_node : *op_stmts) {
    DoEach(op_stmt_node);
  }
}

template <typename DoEachT>
void VisitEachTensor(const m_expr::OpStmt& op, const DoEachT& DoEach) {
  const auto& [op, inputs, outputs] = op.tuple();
  for (const auto& input : inputs.value()) {
    DoEach(input);
  }
  for (const auto& output : outputs.value()) {
    DoEach(output);
  }
}

void CollectTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret);

void CollectTensorIndexIterators(const Undefined& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  LOG(FATAL) << "Not Implemented";
}

void CollectTensorIndexIterators(const equation::IterVar& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  ret->insert(tensor_index_expr);
}

void CollectTensorIndexIterators(const List<equation::Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  for (const auto& value : *tensor_index_expr) {
    CollectTensorIndexIterators(value, ret);
  }
}

void CollectTensorIndexIterators(
    const equation::IndexDot<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(
    const equation::IndexUnDot<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantAdd<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantDiv<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ConstantMod<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const equation::ListGetItem<equation::Value, equation::Constant>&
        tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetList(), ret);
}

void CollectTensorIndexIterators(
    const equation::PtrGetItem<equation::Value>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg1(), ret);
}

void CollectTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  std::visit(
      [&](auto&& impl) { CollectTensorIndexIterators(std::move(impl), ret); },
      tensor_index_expr.variant());
}

std::unordered_set<equation::IterVar> GetTensorIndexIterators(
    const m_expr::TensorIndexExpr& tensor_index_expr) {
  std::unordered_set<equation::IterVar> ret;

  CollectTensorIndexIterators(tensor_index_expr, &ret);

  return ret;
}

LoopIterators GetLeftAlignedSdIterators(
    const std::unordered_set<equation::IterVar>& tensor_index_loop_iters,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor) {
  const auto& Used = [&](const equation::IterVar& iter_var) {
    return tensor_index_loop_iters.count(iter_var) != 0;
  };

  const auto& IsSpatial = [&](const equation::IterVar& iter_var) {
    return IsSpatial(GetLoopDescriptor(iter_var).GetLoopType());
  };

  LoopIterators ret{loop_iters->begin(), loop_iters->end()};
  for (int i = ret->size() - 1; i >= 0; --i) {
    if (Used(ret->at(i)) || IsSpatial(ret->at(i))) {
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
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& tensor_index_loop_iters =
      GetTensorIndexIterators(TensorIndexExpr4Tensor(tensor));

  return GetLeftAlignedSdIterators(
      tensor_index_loop_iters, loop_iters, GetLoopDescriptor);
}

// Schedule Iterator always be aligned
LoopIterators MergeLoopIterators(const LoopIterators& op_loop_iterators,
                                 const LoopIterators& tensor_loop_iterators) {
  return op_loop_iterators->size() > tensor_loop_iterators->size()
             ? op_loop_iterators
             : tensor_loop_iterators;
}

LoopIterators GenerateLoopIterators(
    const m_expr::OpStmt& op,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor,
    std::unordered_map<m_expr::Tensor, LoopIterators>* tensor2loop_iters) {
  LoopIterators op_loop_iterators;
  VisitEachTensor(op, [&](const m_expr::Tensor& tensor) {
    LoopIterators tensor_loop_iterators = GetTensorLoopIterators(
        tensor, loop_iters, GetLoopDescriptor, TensorIndexExpr4Tensor);
    const auto& iter =
        tensor2loop_iters->emplace(tensor, tensor_loop_iterators).first;
    CHECK(*iter == tensor_loop_iterators);
    op_loop_iterators =
        MergeLoopIterators(op_loop_iterators, tensor_loop_iterators);
  });

  return op_loop_iterators;
}

std::pair<std::function<const LoopIterators&(const m_expr::OpStmt&)>,
          std::function<const LoopIterators&(const m_expr::Tensor&)>>
MakeGetterSdIters(
    const List<m_expr::OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor) {
  using Op2ItersCache = std::unordered_map<const m_expr::OpStmt, LoopIterators>;
  const auto& op2loop_iters = std::make_shared<Op2ItersCache>();
  using Tensor2ItersCache = std::unordered_map<m_expr::Tensor, LoopIterators>;
  const auto& tensor2loop_iters = std::make_shared<Tensor2ItersCache>();

  VisitEachOpStmt(op_stmts, [&](const m_expr::OpStmt& op) {
    const auto& value = GenerateLoopIterators(op,
                                              loop_iters,
                                              GetLoopDescriptor,
                                              TensorIndexExpr4Tensor,
                                              tensor2loop_iters.get());
    CHECK(op2loop_iters->emplace(op, value).second);
  });

  return std::pair{[op2loop_iters](const m_expr::OpStmt& op) {
                     return op2loop_iters->at(op);
                   },
                   [tensor2loop_iters](const m_expr::Tensor& tensor) {
                     return tensor2loop_iters->at(tensor);
                   }};
}

MapIrList GenerateOpClusters(
    const List<m_expr::OpStmt>& op_stmts,
    const std::function<const LoopIterators&(const m_expr::OpStmt&)>&
        SdIters4Op,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor) {
  MapIrList map_irs{};

  VisitEachOpStmt(op_stmts, [&](const auto& op_stmt_node) {
    map_irs.emplace_back(MapIr{op_stmt_node, SdIters4Op(op_stmt_node)});
  });

  return map_irs;
}

// Reorder and merge
bool MergePrevToNext4LoopFuse(
    MapIrList* map_irs,
    const std::function<const LoopIterators&(const m_expr::Tensor&)>&
        SdIterators4Tensor) {
  std::size_t merge_count = 0;

  const auto& IsMergable = [&](const auto& src, const auto& dst) {
    return src.IsMergableTo(dst, SdIterators4Tensor);
  };

  const auto& HasReadWriteDependence = [&](const auto& src, const auto& dst) {
    return src.HasReadWriteDependence(dst);
  };

  const auto& MergeSrcToDst = [&](const auto& src, const auto& dst) {
    return src.MergeThisToThat(dst);
  };

  const auto& CouldPrevMergedToNext = [&](auto iter) {
    if (iter == map_irs->begin() || iter == std::prev(map_irs->end())) {
      return false;
    } else {
      auto me = iter;
      auto prev = std::prev(iter);
      auto next = std::next(iter);
      return IsMergable(*prev, *next) && !IsMergable(*me, *next) &&
             !HasReadWriteDependence(*prev, *me);
    }
  };

  const auto& MergePrevToNext = [&](auto iter) {
    CHECK(iter != std::prev(map_irs->end()));
    CHECK(iter != map_irs->begin());
    // Reorder process: prev, iter, next -> iter, merge(prev, next)
    MergeSrcToDst(*std::prev(iter), *std::next(iter));
  };

  for (auto iter = map_irs->begin(); iter != map_irs->end();) {
    if (CouldPrevMergedToNext(iter)) {
      MergePrevToNext(iter);
      --iter;
      iter = map_irs->erase(iter);
      ++merge_count;
    } else {
      ++iter;
    }
  }
  return merge_count > 0;
}

bool MergeNextOrPrev4LoopFuse(
    MapIrList* map_irs,
    const std::function<const LoopIterators&(const m_expr::Tensor&)>&
        SdIterators4Tensor) {
  std::size_t merge_count = 0;

  const auto& IsMergable = [&](const auto& src, const auto& dst) {
    return src.IsMergableTo(dst, SdIterators4Tensor);
  };

  const auto& MergeSrcToDst = [&](const auto& src, const auto& dst) {
    return src.MergeThisToThat(dst);
  };

  const auto& CouldThisMergedToNext = [&](auto iter) {
    if (iter == std::prev(map_irs->end())) {
      return false;
    } else {
      return IsMergable(*iter, *std::next(iter));
    }
  };

  const auto& CouldPrevMergedToThis = [&](auto iter) {
    if (iter == map_irs->begin()) {
      return false;
    } else {
      return CouldThisMergedToNext(std::prev(iter));
    }
  };

  const auto& MergeThisToNext = [&](auto iter) {
    CHECK(iter != std::prev(map_irs->end()));
    MergeSrcToDst(*iter, *std::next(iter));
  };

  const auto& MergePrevToThis = [&](auto iter) {
    CHECK(iter != map_irs->begin());
    MergeSrcToDst(*std::prev(iter), *iter);
  };

  for (auto iter = map_irs->begin(); iter != map_irs->end();) {
    if (CouldThisMergedToNext(iter)) {
      MergeThisToNext(iter);
      iter = map_irs->erase(iter);
      ++merge_count;
    } else if (CouldPrevMergedToThis(iter)) {
      MergePrevToThis(iter);
      --iter;
      iter = map_irs->erase(iter);
      ++merge_count;
    } else {
      ++iter;
    }
  }
  return merge_count > 0;
}

MapIrList ReorderAndMergeOpCluster4LoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const std::function<const LoopIterators&(const m_expr::OpStmt&)>&
        SdIters4Op,
    const std::function<const LoopIterators&(const m_expr::Tensor&)>&
        SdIters4Tensor,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor) {
  MapIrList map_irs =
      GenerateOpClusters(op_stmts, SdIters4Op, GetLoopDescriptor);

  // Reorder and merge
  while (MergePrevToNext4LoopFuse(&map_irs, SdIters4Tensor)) {
  }
  // Merge
  while (MergeNextOrPrev4LoopFuse(&map_irs, SdIters4Tensor)) {
  }

  return map_irs;
}

MapIrList GenerateClusterOpsForLoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor) {
  const auto& [SdIters4Op, SdIters4Tensor] = MakeGetterSdIters(
      op_stmts, loop_iters, GetLoopDescriptor, TensorIndexExpr4Tensor);

  return ReorderAndMergeOpCluster4LoopFuse(
      op_stmts, SdIters4Op, SdIters4Tensor, GetLoopDescriptor);
}

namespace {

std::unordered_map<const equation::Variable, equation::Value>
MakeAnchorIndex2Ok(const equation::Index& anchor_index) {
  return {{anchor_index, Ok{}}};
}

}  // namespace

bool LocalEquationsSolvable(
    const equation::GraphView& graph_view,
    const equation::Index& anchor_index,
    const equation::FakeOpPlaceHolder& fake_op_placeholder) {
  const auto& init_var2value = MakeAnchorIndex2Ok(anchor_index);
  equation::IndexExprInferContext ctx{init_var2value};
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
         match{[&](const Undefined&) -> m_expr::Tensor {
                 LOG(FATAL) << "position not found";
               },
               [&](const tIn<std::size_t>& pos) -> m_expr::Tensor {
                 return in_args.value()->at(pos.value());
               },
               [&](const tOut<std::size_t>& pos) -> m_expr::Tensor {
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
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
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
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
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
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
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
