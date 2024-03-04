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

#include "paddle/cinn/adt/map_expr.h"

namespace cinn::adt {

using LoopIterators = List<Iterator>;

}

namespace cinn::adt {

class MapIr final {
 public:
  MapIr(const List<OpStmt>& op_stmts, const LoopIterators& loop_iterators)
      : op_stmts_{op_stmts}, loop_iterators_(loop_iterators) {}
  MapIr(const MapIr&) = default;
  MapIr(MapIr&&) = default;

  const List<OpStmt>& op_stmts() const { return op_stmts_; }

  const LoopIterators& loop_iterators() const { return loop_iterators_; }

 private:
  List<OpStmt> op_stmts_;
  LoopIterators loop_iterators_;
};

using MapIrList = List<MapIr>;

MapIrList GenerateMapIrListForLoopFuse(
    const List<OpStmt>& op_stmts,
    const LoopIterators& loop_iters,
    const std::function<TensorIndexExpr(const Tensor&)>&
        TensorIndexExpr4Tensor);

void CollectTensorIndexIterators(const TensorIndexExpr& tensor_index_expr,
                                 std::unordered_set<Iterator>* ret);

}  // namespace cinn::adt
