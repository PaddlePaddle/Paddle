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

#pragma once

#include <list>

#include "paddle/cinn/adt/m_ir.h"

namespace cinn::adt {

using ScheduleIterators = List<equation::IterVar>;

}

namespace cinn::adt::op_cluster {

DEFINE_ADT_TAG(tAsOutput);
DEFINE_ADT_TAG(tBreak);

class SdOpStmtNodes final {
 public:
  SdOpStmtNodes(const m_ir::OpStmtNode& op, const ScheduleIterators& sd_iters)
      : ops_{ops}, sd_iters_(sd_iters) {}

  const std::list<m_ir::OpStmtNode>& ops() const { return ops_; }

  const m_expr::ScheduleIterators& sd_iters() const { return sd_iters_; }

  void AddOpStmtNode(
      const m_ir::OpStmtNode& op_stmt_node,
      const std::function<const ScheduleIterators&(const m_ir::OpStmtNode&)>&
          SdIters4Op,
      const std::function<const m_expr::SchedulePolicy&(
          const equation::IterVar&)>& GetSchedulePolicy) {
    ADT_TODO();
  }

  bool IsMergableTo(
      const SdOpStmtNodes& that,
      const std::function<const ScheduleIterators&(const m_ir::Tensor&)>&
          SdIterators4Tensor) const;

  bool HasReadWriteDependence(const SdOpStmtNodes& that) const;

  void MergeThisToThat(const SdOpStmtNodes& that);

 private:
  template <typename DoEachT>
  tBreak<bool> AggregateTensorPair(const SdOpStmtNodes& that,
                                   const DoEachT& DoEach) const;

  template <typename DoEachT>
  void VisitEachTensor(const DoEachT& DoEach) const;

  template <typename DoEachT>
  tBreak<bool> ForEachTensor(const DoEachT& DoEach) const;

  std::unordered_map<m_ir::Tensor, tAsOutput<bool>> GetTensor2AsOutput() const;

  std::list<m_ir::OpStmtNode> ops_;
  ScheduleIterators sd_iters_;
};

using OpClusters = std::list<op_cluster::SdOpStmtNodes>;

OpClusters GenerateClusterOpsForLoopFuse(
    const m_ir::MapIR& map_ir,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleType,
    const std::function<TensorIndexExpr(const m_ir::Tensor&)>&
        GetTensorIndexes);

}  // namespace cinn::adt::op_cluster
