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

#include "paddle/cinn/adt/direction_equation_generator.h"
#include "paddle/cinn/adt/equation_function.h"
#include "paddle/cinn/adt/map_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"

namespace cinn::adt {

namespace config {
class NaiveOpEquationContext;
}

class NaiveBidirectionEquationGenerator : public DirectionEquationGenerator {
 public:
  using EquationCtx4OpStmtT =
      std::function<std::shared_ptr<config::NaiveOpEquationContext>(
          const OpStmt&)>;

  NaiveBidirectionEquationGenerator(const NaiveBidirectionEquationGenerator&) =
      delete;
  NaiveBidirectionEquationGenerator(NaiveBidirectionEquationGenerator&&) =
      delete;

  NaiveBidirectionEquationGenerator(
      const List<OpStmt>& op_stmts,
      const EquationCtx4OpStmtT& EquationCtx4OpStmt)
      : op_stmts_(op_stmts), EquationCtx4OpStmt_(EquationCtx4OpStmt) {
    Init();
  }

  Equations GetDirectionEquations() const override { return equations_; }

  std::function<const OpStmt*(const FakeOpPlaceHolder&)>
  MakeGetterOpStmt4OpPlaceHolder() const override;

  std::optional<Index> OutMsgIndex4InMsgIndex(
      const Index& index) const override {
    const auto& iter = in_msg_index2out_msg_index_.find(index);
    if (iter == in_msg_index2out_msg_index_.end()) {
      return std::nullopt;
    } else {
      return iter->second;
    }
  }

  const List<OpStmt>& op_stmts() const { return op_stmts_; }

  const EquationCtx4OpStmtT& EquationCtx4OpStmt() const {
    return EquationCtx4OpStmt_;
  }

  const Equations& equations() const { return equations_; }

 private:
  void InitInMsgIndex2OutMsgIndex();
  void InitEquations();

  void Init() {
    InitInMsgIndex2OutMsgIndex();
    InitEquations();
  }

 protected:
  List<OpStmt> op_stmts_;
  EquationCtx4OpStmtT EquationCtx4OpStmt_;
  Equations equations_;
  List<FakeOpPlaceHolder> fake_op_placeholders_;
  std::unordered_map<Index, Index> in_msg_index2out_msg_index_;
};

}  // namespace cinn::adt
