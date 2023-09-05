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

#include "paddle/cinn/adt/map_ir_loop_fuse.h"

#include <functional>

// [(ScheduleDescriptor, [OpNode])]
// using Schedule4MergedOps = List<Tuple<ScheduleIterators, List<m_ir::Op>>>;

namespace cinn::adt {

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

ScheduleIterators GetTensorScheduleIterators(
    const cinn::hlir::framework::NodeData* tensor,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleDescriptor,
    const std::function<TensorIndexExpr(
        const cinn::hlir::framework::NodeData*)>& GetTensorIndexes) {
  ADT_TODO();
}

ScheduleIterators MergeScheduleIterators(
    const ScheduleIterators& op_schedule_iterators,
    const ScheduleIterators& tensor_schedule_iterators) {
  ADT_TODO();
}

ScheduleIterators GenerateScheduleIterators(
    const OpStmtNode& op,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleDescriptor,
    const std::function<TensorIndexExpr(
        const cinn::hlir::framework::NodeData*)>& GetTensorIndexes) {
  ScheduleIterators op_schedule_iterators;
  VisitEachTensor(op, [&](const cinn::hlir::framework::NodeData* tensor) {
    ScheduleIterators tensor_schedule_iterators = GetTensorScheduleIterators(
        tensor, sd_iters, GetScheduleDescriptor, GetTensorIndexes);
    op_schedule_iterators = MergeScheduleIterators(op_schedule_iterators,
                                                   tensor_schedule_iterators);
  });

  return op_schedule_iterators;
}

std::function<const ScheduleIterators&(const cinn::hlir::framework::Node*)>
MakeGetterSdIters4Op(
    const m_ir::MapIR& map_ir,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleDescriptor,
    const std::function<TensorIndexExpr(
        const cinn::hlir::framework::NodeData*)>& GetTensorIndexes) {
  using Cache =
      std::unordered_map<const cinn::hlir::framework::Node*, ScheduleIterators>;
  const auto& op2sd_iters = std::make_shared<Cache>();

  VisitEachOpStmtNode(map_ir, [&](const OpStmtNode& op) {
    CHECK(op2sd_iters
              ->emplace(
                  GetIteratorOpKey(op),
                  GenerateScheduleIterators(
                      op, sd_iters, GetScheduleDescriptor, GetTensorIndexes))
              .second);
  });

  return [op2sd_iters](const cinn::hlir::framework::Node* op) {
    return op2sd_iters->at(op);
  };
}

Schedule4MergedOps GenerateClusterOpsForLoopFuse(
    const m_ir::MapIR& map_ir,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleDescriptor,
    const std::function<TensorIndexExpr(
        const cinn::hlir::framework::NodeData*)>& GetTensorIndexes) {
  const auto& SdIters4Op = MakeGetterSdIters4Op(
      map_ir, sd_iters, GetScheduleDescriptor, GetTensorIndexes);

  const auto& reordered_ops =
      Reorder4OpCluster(map_ir, SdIters4Op, GetScheduleDescriptor);

  return GenerateClusterOpsForLoopFuse(
      reordered_ops, sd_iters, GetScheduleDescriptor, SdIters4Op);
}

}  // namespace cinn::adt
