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

#include <algorithm>

#include "paddle/cinn/adt/adapter.h"
#include "paddle/cinn/adt/m_expr.h"
#include "paddle/cinn/adt/naive_op_equation_context.h"

namespace cinn::adt::equation::config {

namespace {

using InBox2OutBox = InMsgBox2OutMsgBox<tOut<tOutMsgBox<OpArgIndexes>>,
                                        tIn<tInMsgBox<OpArgIndexes>>>;

template <typename DoEachEquationT>
Equations TransformEquations(const Equations& origin_equations,
                             const DoEachEquationT& HandleInMsgBox2OutMsgBox,
                             const DoEachEquationT& HandleOther) {
  Equations equations{};
  for (const auto& origin_equation : *origin_equations) {
    if (origin_equation.Has<InBox2OutBox>()) {
      equations->emplace_back(HandleInMsgBox2OutMsgBox(origin_equation));
    } else {
      equations->emplace_back(HandleOther(origin_equation));
    }
  }
  return equations;
}

List<Index> GetNonErasedIndexes(
    const List<Index>& origin_indexes,
    const std::vector<equation::Index>& erased_indexes) {
  List<Index> indexes{};
  for (const auto& index : *origin_indexes) {
    if (std::find(index, erased_indexes.begin(), erased_indexes.end()) ==
        erased_indexes.end()) {
      indexes->emplace_back(index);
    }
  }
  return indexes;
}

Equation EraseOutMsgBoxIndexes(
    const Equation& equation,
    const std::vector<equation::Index>& erased_output_tensor_indexes) {
  const auto& in_msg_box2out_msg_box = equation.Get<InBox2OutBox>();
  const auto& [op_placeholder, out_box_indexes, in_box_indexes] =
      in_msg_box2out_msg_box.tuple();
  const auto& [out_box_in_indexes, out_box_out_indexes] =
      out_box_indexes.value().value().tuple();
  const auto& non_erased_out_box_out_indexes = GetNonErasedIndexes(
      out_box_out_indexes.value(), erased_output_tensor_indexes);
  return InBox2OutBox{op_placeholder,
                      tOut<tOutMsgBox<OpArgIndexes>>{OpArgIndexes{
                          out_box_in_indexes, non_erased_out_box_out_indexes}},
                      in_box_indexes};
}

}  // namespace

void NativeOpEquationContext::EraseOutMsgBoxIndexes(
    const std::vector<equation::Index>& erased_output_tensor_indexes) {
  const auto& Identity = [](const auto& equation) { return equation };
  const auto& Erase = [&](const auto& equation) {
    return EraseOutMsgBoxIndexes(equation, erased_output_tensor_indexes);
  };
  equations_ = TransformEquations(equations_, Erase, Identity);
}

std::vector<std::uint64_t> MakeTensorRanks(const List<m_expr::Arg>& arg_lists) {
  std::vector<std::uint64_t> ret;
  for (const auto& arg : *arg_lists) {
    CHECK(arg.Has<adapter::Tensor>());
    ret.push_back(arg.Get<adapter::Tensor>().GetRank());
  }
  return ret;
}

void GenerateOpEquations(const m_expr::OpStmt& op_stmt,
                         equation::config::NativeOpEquationContext* ctx) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  CHECK(op.Has<const hlir::framework::Node*>());
  const hlir::framework::Node* op_node = op.Get<const hlir::framework::Node*>();

  using GenerateEquationFunc =
      std::function<void(equation::config::NativeOpEquationContext * ctx)>;

  const auto& generate_equations =
      hlir::framework::Operator::GetAttrs<GenerateEquationFunc>(
          "generate_equations");
  const auto& iter = generate_equations.find(op_node->op());
  CHECK(iter != generate_equations.end());
  iter->second(ctx);
}

std::shared_ptr<equation::config::NativeOpEquationContext>
MakeContextAndGenerateEquations(const m_expr::OpStmt& op_stmt) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  const auto& ctx = std::make_shared<equation::config::NativeOpEquationContext>(
      MakeTensorRanks(inputs.value()), MakeTensorRanks(outputs.value()));

  GenerateOpEquations(op_stmt, ctx.get());

  return ctx;
}

std::function<std::shared_ptr<equation::config::NativeOpEquationContext>(
    const m_expr::OpStmt&)>
GenerateContext4LocalOpStmt(const List<m_expr::OpStmt>& op_stmts) {
  using OpStmt2EquationContext = std::unordered_map<
      m_expr::OpStmt,
      std::shared_ptr<equation::config::NativeOpEquationContext>>;
  const auto& op_stmt2equation_ctx = std::make_shared<OpStmt2EquationContext>();

  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = MakeContextAndGenerateEquations(op_stmt);
    CHECK(op_stmt2equation_ctx->emplace(op_stmt, ctx).second);
  }

  return [op_stmt2equation_ctx](const auto& op_stmt) {
    return op_stmt2equation_ctx->at(op_stmt);
  };
}

}  // namespace cinn::adt::equation::config
