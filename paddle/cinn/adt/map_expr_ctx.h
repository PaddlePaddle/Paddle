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

#include <memory>
#include <unordered_map>

#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/ir/lowered_func.h"
#include "paddle/cinn/ir/utils/ir_copy.h"
#include "paddle/pir/core/operation.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace cinn::adt {

class MapExprCtx final {
 public:
  using Node2LoweredFuncs =
      std::unordered_map<::pir::Operation*, std::vector<ir::LoweredFunc>>;

  MapExprCtx(const MapExprCtx&) = delete;
  MapExprCtx(MapExprCtx&&) = delete;

  explicit MapExprCtx(
      const MapExpr& map_expr,
      const std::unordered_map<SymbolicDim, ::pir::shape::SymbolicDimOp>&
          map_expr_symbolic2dialect_symbolic)
      : map_expr_(map_expr),
        map_expr_symbolic2dialect_symbolic_(
            map_expr_symbolic2dialect_symbolic) {}

  const MapExpr& map_expr() const { return map_expr_; }

  void UpdateOpLoweredFuncKey(
      ::pir::Operation* node,
      const std::vector<ir::LoweredFunc>& lowered_funcs) {
    Node2LoweredFuncs* map = &node2lowered_funcs_;
    CHECK(map->emplace(node, ir::ir_utils::IRCopy(lowered_funcs)).second);
  }

  const Node2LoweredFuncs& node2lowered_funcs() const {
    return node2lowered_funcs_;
  }

  const std::unordered_map<SymbolicDim, ::pir::shape::SymbolicDimOp>&
  map_expr_symbolic2dialect_symbolic() const {
    return map_expr_symbolic2dialect_symbolic_;
  }

 private:
  const MapExpr map_expr_;
  Node2LoweredFuncs node2lowered_funcs_;
  std::unordered_map<SymbolicDim, ::pir::shape::SymbolicDimOp>
      map_expr_symbolic2dialect_symbolic_;
};

}  // namespace cinn::adt
