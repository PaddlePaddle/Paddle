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
  MapIr(const List<m_expr::OpStmt>& op_stmts,
        const List<LoopIterators>& loop_iters_list)
      : op_stmts_{op_stmts}, loop_iters_list_(loop_iters_list) {}
  MapIr(const MapIr&) = default;
  MapIr(MapIr&&) = default;

  const List<m_expr::OpStmt>& op_stmts() const { return op_stmts_; }

  const List<LoopIterators>& loop_iters_list() const {
    return loop_iters_list_;
  }

 private:
  List<m_expr::OpStmt> op_stmts_;
  List<LoopIterators> loop_iters_list_;
};

using MapIrList = List<MapIr>;

MapIrList GenerateMapIrListForLoopFuse(
    const List<m_expr::OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<const LoopDescriptor&(const equation::IterVar&)>&
        GetLoopDescriptor,
    const std::function<const m_expr::TensorIndexExpr&(const m_expr::Tensor&)>&
        TensorIndexExpr4Tensor)

}  // namespace cinn::adt::m_ir
