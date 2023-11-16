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

#include <unordered_map>

#include "paddle/cinn/adt/equation_function_constants_provider.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"

namespace cinn::adt {

class NaiveEquationFunctionConstantsProvider final
    : public EquationFunctionConstantsProvider {
 public:
  using EquationCtx4OpStmtT =
      std::function<std::shared_ptr<config::NaiveOpEquationContext>(
          const OpStmt&)>;

  NaiveEquationFunctionConstantsProvider(
      const List<OpStmt>& op_stmts,
      const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    Init(op_stmts, EquationCtx4OpStmt);
  }

  NaiveEquationFunctionConstantsProvider(
      const NaiveEquationFunctionConstantsProvider&) = delete;
  NaiveEquationFunctionConstantsProvider(
      NaiveEquationFunctionConstantsProvider&&) = delete;

  Constant GetDimSize(const Dim& dim) const override {
    const auto& iter = dim2constant_.find(dim);
    CHECK(iter != dim2constant_.end());
    return iter->second;
  }

  bool AddDim(const Dim& dim, const Constant& dim_value) override {
    return dim2constant_.emplace(dim, dim_value).second;
  }

 private:
  void Init(const List<OpStmt>& op_stmts,
            const EquationCtx4OpStmtT& EquationCtx4OpStmt) {
    for (const auto& op_stmt : *op_stmts) {
      const auto& ctx = EquationCtx4OpStmt(op_stmt);
      ctx->VisitEachArgPos(
          [&](bool is_out, std::size_t arg_idx, std::size_t axis) {
            const Dim& dim = ctx->GetDim(is_out, arg_idx, axis);
            const Constant& constant = ctx->GetDimSize(is_out, arg_idx, axis);
            CHECK(dim2constant_.emplace(dim, constant).second);
          });
    }
  }

  std::unordered_map<Dim, const Constant> dim2constant_;
};

}  // namespace cinn::adt
