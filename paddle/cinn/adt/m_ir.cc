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
#include "paddle/cinn/adt/m_ir.h"

namespace cinn::adt::m_ir {

template <typename DoEachT>
void MapIR::VisitEachTensor(const DoEachT& DoEach) const {
  ForEachTensor([&](const auto& tensor, const auto& as_output) {
    DoEach(tensor, as_output);
    return tBreak{false};
  });
}

template <typename DoEachT>
tBreak<bool> MapIR::ForEachTensor(const DoEachT& DoEach) const {
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

std::unordered_map<m_expr::Tensor, tAsOutput<bool>> MapIR::GetTensor2AsOutput()
    const {
  std::unordered_map<m_expr::Tensor, tAsOutput<bool>> ret{};

  VisitEachTensor([&](const m_expr::Tensor& tensor, tAsOutput<bool> as_output) {
    ret[tensor] = ret[tensor].value() || as_output.value();
  });

  return ret;
}

template <typename DoEachT>
tBreak<bool> MapIR::AggregateTensorPair(const MapIR& that,
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

bool MapIR::IsMergableTo(
    const MapIR& that,
    const std::function<const ScheduleIterators&(const m_expr::Tensor&)>&
        SdIterators4Tensor) const {
  // TODO(Hongyu Jia): Refact to support complicated cases
  if (that.sd_iters()->size() < this->sd_iters()->size()) {
    return false;
  }

  const auto& CheckBroadcast = [&](const auto& tensor) {
    return SdIterators4Tensor(tensor).size() < that.sd_iters().size();
  };

  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };
  bool mergable = true;
  const auto& UpdataMergable = [&](const auto& tensor,
                                   tAsOutput<bool> this_as_output,
                                   tAsOutput<bool> that_as_output) {
    if (CheckBroadcast(tensor) && CheckWrite(this_as_output, that_as_output)) {
      mergable = false;
      return tBreak{true};
    } else {
      return tBreak{false};
    }
  };
  AggregateTensorPair(that, UpdataMergable);
  return mergable;
}

bool MapIR::HasReadWriteDependence(const MapIR& that) const {
  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };

  bool has_read_write_dependence = false;

  AggregateTensorPair(that,
                      [&](const auto& tensor,
                          tAsOutput<bool> this_as_output,
                          tAsOutput<bool> that_as_output) {
                        if (CheckWrite(this_as_output, that_as_output)) {
                          has_read_write_dependence = true;
                          return tBreak{true};
                        } else {
                          return tBreak{false};
                        }
                      });

  return has_read_write_dependence;
}

void MapIR::MergeThisToThat(const MapIR& that) {
  CHECK_GE(that.sd_iters()->size(), this->sd_iters()->size());
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

ScheduleIterators GetLeftAlignedSdIterators(
    const std::unordered_set<equation::IterVar>& tensor_index_sd_iters,
    const ScheduleIterators& sd_iters,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy) {
  const auto& Used = [&](const equation::IterVar& iter_var) {
    return tensor_index_sd_iters.count(iter_var) != 0;
  };

  const auto& IsSpatial = [&](const equation::IterVar& iter_var) {
    return IsSpatial(GetSchedulePolicy(iter_var).GetScheduleType());
  };

  ScheduleIterators ret{sd_iters->begin(), sd_iters->end()};
  for (int i = ret->size() - 1; i >= 0; --i) {
    if (Used(ret->at(i)) || IsSpatial(ret->at(i))) {
      break;
    } else {
      ret->resize(i);
    }
  }
  return ret;
}

ScheduleIterators GetTensorScheduleIterators(
    const m_expr::Tensor& tensor,
    const ScheduleIterators& sd_iters,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes) {
  const auto& tensor_index_sd_iters =
      GetTensorIndexIterators(GetTensorIndexes(tensor));

  return GetLeftAlignedSdIterators(
      tensor_index_sd_iters, sd_iters, GetSchedulePolicy);
}

// Schedule Iterator always be aligned
ScheduleIterators MergeScheduleIterators(
    const ScheduleIterators& op_schedule_iterators,
    const ScheduleIterators& tensor_schedule_iterators) {
  return op_schedule_iterators->size() > tensor_schedule_iterators->size()
             ? op_schedule_iterators
             : tensor_schedule_iterators;
}

ScheduleIterators GenerateScheduleIterators(
    const m_expr::OpStmt& op,
    const ScheduleIterators& sd_iters,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes,
    std::unordered_map<m_expr::Tensor, ScheduleIterators>* tensor2sd_iters) {
  ScheduleIterators op_schedule_iterators;
  VisitEachTensor(op, [&](const m_expr::Tensor& tensor) {
    ScheduleIterators tensor_schedule_iterators = GetTensorScheduleIterators(
        tensor, sd_iters, GetSchedulePolicy, GetTensorIndexes);
    const auto& iter =
        tensor2sd_iters->emplace(tensor, tensor_schedule_iterators).first;
    CHECK(*iter == tensor_schedule_iterators);
    op_schedule_iterators = MergeScheduleIterators(op_schedule_iterators,
                                                   tensor_schedule_iterators);
  });

  return op_schedule_iterators;
}

std::pair<std::function<const ScheduleIterators&(const m_expr::OpStmt&)>,
          std::function<const ScheduleIterators&(const m_expr::Tensor&)>>
MakeGetterSdIters(
    const List<m_expr::OpStmt>& op_stmts,
    const ScheduleIterators& sd_iters,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes) {
  using Op2ItersCache =
      std::unordered_map<const m_expr::OpStmt, ScheduleIterators>;
  const auto& op2sd_iters = std::make_shared<Op2ItersCache>();
  using Tensor2ItersCache =
      std::unordered_map<m_expr::Tensor, ScheduleIterators>;
  const auto& tensor2sd_iters = std::make_shared<Tensor2ItersCache>();

  VisitEachOpStmt(op_stmts, [&](const m_expr::OpStmt& op) {
    const auto& value = GenerateScheduleIterators(op,
                                                  sd_iters,
                                                  GetSchedulePolicy,
                                                  GetTensorIndexes,
                                                  tensor2sd_iters.get());
    CHECK(op2sd_iters->emplace(op, value).second);
  });

  return std::pair{
      [op2sd_iters](const m_expr::OpStmt& op) { return op2sd_iters->at(op); },
      [tensor2sd_iters](const m_expr::Tensor& tensor) {
        return tensor2sd_iters->at(tensor);
      }};
}

MapIRList GenerateOpClusters(
    const List<m_expr::OpStmt>& op_stmts,
    const std::function<const ScheduleIterators&(const m_expr::OpStmt&)>&
        SdIters4Op,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy) {
  MapIRList map_irs{};

  VisitEachOpStmt(op_stmts, [&](const auto& op_stmt_node) {
    map_irs.emplace_back(MapIR{op_stmt_node, SdIters4Op(op_stmt_node)});
  });

  return map_irs;
}

// Reorder and merge
bool MergePrevToNext4LoopFuse(
    MapIRList* map_irs,
    const std::function<const ScheduleIterators&(const m_expr::Tensor&)>&
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
      iter = op_cluster->erase(iter);
      ++merge_count;
    } else {
      ++iter;
    }
  }
  return merge_count > 0;
}

bool MergeNextOrPrev4LoopFuse(
    MapIRList* map_irs,
    const std::function<const ScheduleIterators&(const m_expr::Tensor&)>&
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
      iter = op_cluster->erase(iter);
      ++merge_count;
    } else if (CouldPrevMergedToThis(iter)) {
      MergePrevToThis(iter);
      --iter;
      iter = op_cluster->erase(iter);
      ++merge_count;
    } else {
      ++iter;
    }
  }
  return merge_count > 0;
}

MapIRList ReorderAndMergeOpCluster4LoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const std::function<const ScheduleIterators&(const m_expr::OpStmt&)>&
        SdIters4Op,
    const std::function<const ScheduleIterators&(const m_expr::Tensor&)>&
        SdIters4Tensor,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy) {
  MapIRList map_irs =
      GenerateOpClusters(op_stmts, SdIters4Op, GetSchedulePolicy);

  // Reorder and merge
  while (MergePrevToNext4LoopFuse(&op_cluster, SdIters4Tensor)) {
  }
  // Merge
  while (MergeNextOrPrev4LoopFuse(&op_cluster, SdIters4Tensor)) {
  }

  return map_irs;
}

MapIRList GenerateClusterOpsForLoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const ScheduleIterators& sd_iters,
    const std::function<const SchedulePolicy&(const equation::IterVar&)>&
        GetSchedulePolicy,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes) {
  const auto& [SdIters4Op, SdIters4Tensor] = MakeGetterSdIters(
      op_stmts, sd_iters, GetSchedulePolicy, GetTensorIndexes);

  return ReorderAndMergeOpCluster4LoopFuse(
      op_stmts, SdIters4Op, SdIters4Tensor, GetSchedulePolicy);
}

}  // namespace cinn::adt::m_ir
