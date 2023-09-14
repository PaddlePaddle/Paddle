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

using LoopIterators = List<equation::IterVar>;

}

namespace cinn::adt::m_ir {

class MapIr final {
 public:
  MapIr(const m_expr::OpStmt& op_stmt, const LoopIterators& loop_iters)
      : op_stmts_{op_stmt}, loop_iters_(loop_iters) {}

  const std::list<m_expr::OpStmt>& op_stmts() const { return op_stmts_; }

  const cinn::adt::LoopIterators& loop_iters() const { return loop_iters_; }

  bool IsMergableTo(
      const MapIr& that,
      const std::function<const LoopIterators&(const m_expr::Tensor&)>&
          SdIterators4Tensor) const;

  bool HasReadWriteDependence(const MapIr& that) const;

  void MergeThisToThat(const MapIr& that);

 private:
  template <typename DoEachT>
  tBreak<bool> AggregateTensorPair(const MapIr& that,
                                   const DoEachT& DoEach) const;

  template <typename DoEachT>
  void VisitEachTensor(const DoEachT& DoEach) const;

  template <typename DoEachT>
  tBreak<bool> ForEachTensor(const DoEachT& DoEach) const;

  std::unordered_map<m_expr::Tensor, tAsOutput<bool>> GetTensor2AsOutput()
      const;

  List<m_expr::OpStmt> op_stmts_;
  List<LoopIterators> loop_iters_list_;
};

using MapIrList = List<MapIr>;

MapIrList GenerateClusterOpsForLoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const cinn::adt::LoopDescriptor&(
        const equation::IterVar&)>& GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        GetTensorIndexes);

}  // namespace cinn::adt::m_ir
