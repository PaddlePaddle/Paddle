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

#include "paddle/cinn/adt/m_ir.h"
#include "paddle/cinn/adt/op_cluster.h"

namespace cinn::adt::op_cluster {

template <typename DoEachT>
void SdOpStmtNodes::VisitEachTensor(const DoEachT& DoEach) const {
  ForEachTensor<tBreak>([&](const auto& tensor, const auto& as_output) {
    DoEach(tensor, as_output);
    return tBreak{false};
  });
}

template <template <typename> class ReturnTag, typename DoEachT>
ReturnTag<bool> SdOpStmtNodes::ForEachTensor(const DoEachT& DoEach) const {
  for (const auto& op_node : ops_) {
    const auto& [op, inputs, outputs] = op_node.tuple();
    for (const auto& input : inputs.value()) {
      const auto& tag = DoEach(input, false);
      if (tag.value()) {
        return tag;
      }
    }
    for (const auto& output : outputs.value()) {
      const auto& tag = DoEach(output, true);
      if (tag.value()) {
        return tag;
      }
    }
  }
  return ReturnTag<bool>{false};
}

std::unordered_map<m_ir::Tensor, tAsOutput<bool>>
SdOpStmtNodes::GetTensor2AsOutput() const {
  std::unordered_map<m_ir::Tensor, tAsOutput<bool>> ret{};

  VisitEachTensor([&](const m_ir::Tensor& tensor, tAsOutput<bool> as_output) {
    ret[tensor] = ret[tensor].value() || as_output.value();
  });

  return ret;
}

template <template <typename> class ReturnTag, typename DoEachT>
ReturnTag<bool> SdOpStmtNodes::AggregateTensorPair(
    const SdOpStmtNodes& that, const DoEachT& DoEach) const {
  auto that_tensor2as_output = that.GetTensor2AsOutput();

  return ForEachTensor<ReturnTag>(
      [&](const auto& this_tenosr, const auto& this_as_output) {
        const auto& pair = that_tensor2as_output.find(this_tensor);
        if (pair != that_tensor2as_output.end()) {
          const auto& [that_tensor, that_as_output] = *pair;
          const auto& tag = DoEach(this_tensor, this_as_output, that_as_output);
          if (tag.value()) {
            return tag;
          }
        }
        return ReturnTag{false};
      });
}

bool SdOpStmtNodes::IsMergableTo(
    const SdOpStmtNodes& that,
    const std::function<const ScheduleIterators&(const m_ir::Tensor&)>&
        SdIterators4Tensor) const {
  if (that.sd_iters().size() < this->sd_iters().size()) {
    return false;
  }

  const auto& CheckBroadcast = [&](const auto& tensor) {
    return SdIterators4Tensor(tensor).size() < that.sd_iters().size();
  };

  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };

  return !AggregateTensorPair<tMergable>(
              that,
              [&](const auto& tensor,
                  tAsOutput<bool> this_as_output,
                  tAsOutput<bool> that_as_output) -> tMergable<bool> {
                return CheckBroadcast(tensor) &&
                       CheckWrite(this_as_output, that_as_output);
              })
              .value();
}

bool SdOpStmtNodes::HasReadWriteDependence(const SdOpStmtNodes& that) const {
  const auto& CheckWrite = [&](const auto& this_as_output,
                               const auto& that_as_output) {
    return this_as_output.value() || that_as_output.value();
  };

  return !AggregateTensorPair<tHasReadWriteDependence>(
              that,
              [&](const auto& tensor,
                  tAsOutput<bool> this_as_output,
                  tAsOutput<bool> that_as_output)
                  -> tHasReadWriteDependence<bool> {
                return CheckWrite(this_as_output, that_as_output);
              })
              .value();
}

void SdOpStmtNodes::MergeThisToThat(const SdOpStmtNodes& that) {
  CHECK_GE(that.sd_iters().size(), this->sd_iters().size());
  that.ops_.splice(that.ops_.begin(), std::move(this->ops_));
}

template <typename DoEachT>
void VisitEachOpStmtNode(const m_ir::MapIR& map_ir, const DoEachT& DoEach) {
  ADT_TODO();
}

const cinn::hlir::framework::Node* GetIteratorOpKey(
    const OpStmtNode& op_stmt_node) {
  // Yifan
  ADT_TODO();
}

template <typename DoEachT>
void VisitEachTensor(const OpStmtNode& op, const DoEachT& DoEach) {
  ADT_TODO();
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret);

void CollectTensorIndexIterators(const Undefined& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  LOG(FATAL) << "Not Implemented";
}

void CollectTensorIndexIterators(const IterVar& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  ret->insert(tensor_index_expr);
}

void CollectTensorIndexIterators(const List<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  for (const auto& value : *tensor_index_expr) {
    CollectTensorIndexIterators(value, ret);
  }
}

void CollectTensorIndexIterators(const IndexDot<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(const IndexUnDot<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetIterators(), ret);
}

void CollectTensorIndexIterators(const ConstantAdd<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(const ConstantDiv<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(const ConstantMod<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg0(), ret);
}

void CollectTensorIndexIterators(
    const ListGetItem<Value, Constant>& tensor_index_expr,
    std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetList(), ret);
}

void CollectTensorIndexIterators(const PtrGetItem<Value>& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  CollectTensorIndexIterators(tensor_index_expr.GetArg1(), ret);
}

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<equation::IterVar>* ret) {
  std::visit(
      [&](auto&& impl) { CollectTensorIndexIterators(std::move(impl), ret); },
      tensor_index_expr.variant());
}

std::unordered_set<equation::IterVar> GetTensorIndexIterators(
    const TensorIndexExpr& tensor_index_expr) {
  std::unordered_set<equation::IterVar> ret;

  CollectTensorIndexIterators(tensor_index_expr, &ret);

  return ret;
}

ScheduleIterators GetLeftAlignedSdIterators(
    const std::unordered_set<equation::IterVar>& tensor_index_sd_iters,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy) {
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
    const m_ir::Tensor& tensor,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy,
    const std::function<TensorIndexExpr(const m_ir::Tensor&)>&
        GetTensorIndexes) {
  const auto& tensor_index_sd_iters =
      GetTensorIndexIterators(GetTensorIndexes(tensor));

  return GetLeftAlignedSdIterators(
      tensor_index_sd_iters, sd_iters, GetSchedulePolicy);
}

ScheduleIterators MergeScheduleIterators(
    const ScheduleIterators& op_schedule_iterators,
    const ScheduleIterators& tensor_schedule_iterators) {
  ADT_TODO();
}

ScheduleIterators GenerateScheduleIterators(
    const OpStmtNode& op,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy,
    const std::function<TensorIndexExpr(const m_ir::Tensor&)>& GetTensorIndexes,
    std::unordered_map<m_ir::Tensor, ScheduleIterators>* tensor2sd_iters) {
  ScheduleIterators op_schedule_iterators;
  VisitEachTensor(op, [&](const m_ir::Tensor& tensor) {
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

std::pair<
    std::function<const ScheduleIterators&(const cinn::hlir::framework::Node*)>,
    std::function<const ScheduleIterators&(const m_ir::Tensor&)>>
MakeGetterSdIters(const m_ir::MapIR& map_ir,
                  const ScheduleIterators& sd_iters,
                  const std::function<const m_expr::SchedulePolicy&(
                      const equation::IterVar&)>& GetSchedulePolicy,
                  const std::function<TensorIndexExpr(const m_ir::Tensor&)>&
                      GetTensorIndexes) {
  using Op2ItersCache =
      std::unordered_map<const cinn::hlir::framework::Node*, ScheduleIterators>;
  const auto& op2sd_iters = std::make_shared<Op2ItersCache>();
  using Tensor2ItersCache = std::unordered_map<m_ir::Tensor, ScheduleIterators>;
  const auto& tensor2sd_iters = std::make_shared<Tensor2ItersCache>();

  VisitEachOpStmtNode(map_ir, [&](const OpStmtNode& op) {
    const auto& key = GetIteratorOpKey(op);
    const auto& value = GenerateScheduleIterators(op,
                                                  sd_iters,
                                                  GetSchedulePolicy,
                                                  GetTensorIndexes,
                                                  tensor2sd_iters.get());
    CHECK(op2sd_iters->emplace(key, value).second);
  });

  return std::pair{[op2sd_iters](const cinn::hlir::framework::Node* op) {
                     return op2sd_iters->at(op);
                   },
                   [tensor2sd_iters](const m_ir::Tensor& tensor) {
                     return tensor2sd_iters->at(tensor);
                   }};
}

OpClusters GenerateOpClusters(
    const m_ir::MapIR& map_ir,
    const std::function<const ScheduleIterators&(
        const cinn::hlir::framework::Node*)>& SdIters4Op,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy) {
  ADT_TODO();
}

// Reorder and merge
bool MergePrevToNext4LoopFuse(
    OpClusters* op_clusters,
    const std::function<const ScheduleIterators&(const m_ir::Tensor&)>&
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
    if (iter == op_clusters->begin() || iter == std::prev(op_clusters->end())) {
      return false;
    } else {
      auto me = iter;
      auto prev = std::prev(iter);
      auto next = std::next(iter);
      return IsMergable(*prev, *next) && !IsMergeable(*me, *next) &&
             !HasReadWriteDependence(*prev, *me);
    }
  };

  const auto& MergePrevToNext = [&](auto iter) {
    CHECK(iter != std::prev(op_clusters->end()));
    CHECK(iter != op_clusters->begin());
    MergeSrcToDst(*std::prev(iter), *std::next(iter));
  };

  for (auto iter = op_clusters->begin(); iter != op_clusters->end();) {
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
    OpClusters* op_clusters,
    const std::function<const ScheduleIterators&(const m_ir::Tensor&)>&
        SdIterators4Tensor) {
  std::size_t merge_count = 0;

  const auto& IsMergable = [&](const auto& src, const auto& dst) {
    return src.IsMergableTo(dst, SdIterators4Tensor);
  };

  const auto& MergeSrcToDst = [&](const auto& src, const auto& dst) {
    return src.MergeThisToThat(dst);
  };

  const auto& CouldThisMergedToNext = [&](auto iter) {
    if (iter == std::prev(op_clusters->end())) {
      return false;
    } else {
      return IsMergable(*iter, *std::next(iter));
    }
  };

  const auto& CouldPrevMergedToThis = [&](auto iter) {
    if (iter == op_clusters->begin()) {
      return false;
    } else {
      return CouldThisMergedToNext(std::prev(iter));
    }
  };

  const auto& MergeThisToNext = [&](auto iter) {
    CHECK(iter != std::prev(op_clusters->end()));
    MergeSrcToDst(*iter, *std::next(iter));
  };

  const auto& MergePrevToThis = [&](auto iter) {
    CHECK(iter != op_clusters->begin());
    MergeSrcToDst(*std::prev(iter), *iter);
  };

  for (auto iter = op_clusters->begin(); iter != op_clusters->end();) {
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

OpClusters ReorderAndMergeOpCluster4LoopFuse(
    const m_ir::MapIR& map_ir,
    const std::function<const ScheduleIterators&(
        const cinn::hlir::framework::Node*)>& SdIters4Op,
    const std::function<const ScheduleIterators&(const m_ir::Tensor&)>&
        SdIters4Tensor,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy) {
  OpClusters op_clusters =
      GenerateOpClusters(map_ir, SdIters4Op, GetSchedulePolicy);

  // Reorder and merge
  while (MergePrevToNext4LoopFuse(&op_cluster, SdIters4Tensor)) {
  }
  // Merge
  while (MergeNextOrPrev4LoopFuse(&op_cluster, SdIters4Tensor)) {
  }

  return op_clusters;
}

OpClusters GenerateClusterOpsForLoopFuse(
    const m_ir::MapIR& map_ir,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::SchedulePolicy&(
        const equation::IterVar&)>& GetSchedulePolicy,
    const std::function<TensorIndexExpr(const m_ir::Tensor&)>&
        GetTensorIndexes) {
  const auto& [SdIters4Op, SdIters4Tensor] =
      MakeGetterSdIters(map_ir, sd_iters, GetSchedulePolicy, GetTensorIndexes);

  return ReorderAndMergeOpCluster4LoopFuse(
      map_ir, SdIters4Op, SdIters4Tensor, GetSchedulePolicy);
}

}  // namespace cinn::adt::op_cluster
