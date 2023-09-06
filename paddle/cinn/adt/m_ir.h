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

#include "paddle/cinn/adt/m_expr.h"

namespace cinn::adt {

using ScheduleIterators = List<equation::IterVar>;

}

namespace cinn::adt::m_ir {

DEFINE_ADT_TAG(tAsOutput);
DEFINE_ADT_TAG(tBreak);

class MapIR final {
 public:
  MapIR(const m_expr::OpStmtNode& op, const ScheduleIterators& sd_iters)
      : ops_{ops}, sd_iters_(sd_iters) {}

  const std::list<m_expr::OpStmtNode>& ops() const { return ops_; }

  const m_expr::ScheduleIterators& sd_iters() const { return sd_iters_; }

  bool IsMergableTo(
      const MapIR& that,
      const std::function<const ScheduleIterators&(const m_expr::Tensor&)>&
          SdIterators4Tensor) const;

  bool HasReadWriteDependence(const MapIR& that) const;

  void MergeThisToThat(const MapIR& that);

 private:
  template <typename DoEachT>
  tBreak<bool> AggregateTensorPair(const MapIR& that,
                                   const DoEachT& DoEach) const;

  template <typename DoEachT>
  void VisitEachTensor(const DoEachT& DoEach) const;

  template <typename DoEachT>
  tBreak<bool> ForEachTensor(const DoEachT& DoEach) const;

  std::unordered_map<m_expr::Tensor, tAsOutput<bool>> GetTensor2AsOutput()
      const;

  std::list<m_expr::OpStmtNode> ops_;
  ScheduleIterators sd_iters_;
};

using MapIRList = std::list<MapIR>;

MapIRList GenerateClusterOpsForLoopFuse(
    const List<m_expr::OpStmtNode>& op_stmt_nodes,
    const ScheduleIterators& sd_iters,
    const std::function<const m_expr::ScheduleDescriptor&(
        const equation::IterVar&)>& GetScheduleType,
    const std::function<TensorIndexExpr(const m_expr::Tensor&)>&
        GetTensorIndexes);

}  // namespace cinn::adt::m_ir
