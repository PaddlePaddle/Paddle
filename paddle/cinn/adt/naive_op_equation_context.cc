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
#include "paddle/cinn/adt/op_arg_pos.h"

#include "glog/logging.h"

namespace cinn::adt::config {

namespace {

using InBox2OutBox = InMsgBox2OutMsgBox<tOut<FakeOpPlaceHolder>,
                                        tOut<tOutMsgBox<OpArgIndexes>>,
                                        tIn<tInMsgBox<OpArgIndexes>>>;

template <typename HandleInMsgBox2OutMsgBoxT, typename HandleOtherT>
Equations TransformEquations(
    const Equations& origin_equations,
    const HandleInMsgBox2OutMsgBoxT& HandleInMsgBox2OutMsgBox,
    const HandleOtherT& HandleOther) {
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

List<Index> GetNonErasedIndexes(const List<Index>& origin_indexes,
                                const std::vector<Index>& erased_indexes) {
  List<Index> indexes{};
  for (const auto& index : *origin_indexes) {
    if (std::find(erased_indexes.begin(), erased_indexes.end(), index) ==
        erased_indexes.end()) {
      indexes->emplace_back(index);
    }
  }
  return indexes;
}

Equation EraseIndexes(const Equation& equation,
                      const std::vector<Index>& erased_output_tensor_indexes) {
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

void NaiveOpEquationContext::EraseOutMsgBoxIndexes(
    const std::vector<Index>& erased_output_tensor_indexes) {
  const auto& Identity = [](const Equation& equation) { return equation; };
  const auto& Erase = [&](const Equation& equation) {
    return EraseIndexes(equation, erased_output_tensor_indexes);
  };
  equations_ = TransformEquations(equations_, Erase, Identity);
}

std::vector<std::uint64_t> MakeTensorRanks(const List<Arg>& arg_lists) {
  std::vector<std::uint64_t> ret;
  for (const auto& arg : *arg_lists) {
    CHECK(arg.Has<adapter::Tensor>());
    ret.push_back(arg.Get<adapter::Tensor>().GetRank());
  }
  return ret;
}

void GenerateOpEquations(const OpStmt& op_stmt,
                         config::NaiveOpEquationContext* ctx) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  CHECK(op.Has<const hlir::framework::Node*>());
  const hlir::framework::Node* op_node = op.Get<const hlir::framework::Node*>();

  using GenerateEquationFunc =
      std::function<void(config::OpEquationContext * ctx)>;

  const auto& generate_equations =
      hlir::framework::Operator::GetAttrs<GenerateEquationFunc>(
          "generate_equations");
  CHECK(generate_equations.Find(op_node->op()));
  generate_equations[op_node->op()](ctx);
}

using GetArgStaticDimT = std::function<std::optional<std::int64_t>(
    std::size_t tensor_idx, std::size_t dim_idx)>;

GetArgStaticDimT MakeGetArgStaticDimT(const List<Tensor>& tensors) {
  return [=](std::size_t tensor_idx,
             std::size_t dim_idx) -> std::optional<std::int64_t> {
    if (tensor_idx >= tensors->size()) {
      return std::nullopt;
    }
    CHECK(tensors->at(tensor_idx).Has<adapter::Tensor>());
    const auto& tensor_shape =
        tensors->at(tensor_idx).Get<adapter::Tensor>().GetShape();
    if (dim_idx >= tensor_shape.size()) {
      return std::nullopt;
    }
    return tensor_shape.at(dim_idx);
  };
}

std::shared_ptr<config::NaiveOpEquationContext> MakeContextAndGenerateEquations(
    const OpStmt& op_stmt) {
  const auto& [op, inputs, outputs] = op_stmt.tuple();
  const auto& ctx = std::make_shared<config::NaiveOpEquationContext>(
      MakeTensorRanks(inputs.value()),
      MakeTensorRanks(outputs.value()),
      MakeGetArgStaticDimT(inputs.value()),
      MakeGetArgStaticDimT(outputs.value()));

  GenerateOpEquations(op_stmt, ctx.get());

  return ctx;
}

std::function<std::shared_ptr<config::NaiveOpEquationContext>(const OpStmt&)>
GenerateContext4LocalOpStmt(const List<OpStmt>& op_stmts) {
  using OpStmt2EquationContext =
      std::unordered_map<OpStmt,
                         std::shared_ptr<config::NaiveOpEquationContext>>;
  const auto& op_stmt2equation_ctx = std::make_shared<OpStmt2EquationContext>();

  for (const auto& op_stmt : *op_stmts) {
    const auto& ctx = MakeContextAndGenerateEquations(op_stmt);
    CHECK(op_stmt2equation_ctx->emplace(op_stmt, ctx).second);
  }

  return [op_stmt2equation_ctx](const auto& op_stmt) {
    return op_stmt2equation_ctx->at(op_stmt);
  };
}

template <typename T0, typename T1>
struct CompLogicalExpr {
  template <typename CompareT>
  static bool Call(const CompareT& Compare, const T0&, const T1&) {
    LOG(FATAL) << "Unimplemented";
  }
};

template <>
struct CompLogicalExpr<std::int64_t, std::int64_t> {
  template <typename CompareT>
  static bool Call(const CompareT& Compare,
                   std::int64_t lhs,
                   std::int64_t rhs) {
    return Compare(lhs, rhs);
  }
};

template <typename CompareT>
bool CalculateLogicalExprImpl(
    const std::tuple<EquationStaticValue, EquationStaticValue>& tuple,
    const CompareT& Compare) {
  const auto& [lhs, rhs] = tuple;
  return std::visit(
      [&](auto&& lhs, auto&& rhs) {
        return CompLogicalExpr<
            std::decay_t<decltype(lhs)>,
            std::decay_t<decltype(rhs)>>::template Call<CompareT>(Compare,
                                                                  lhs,
                                                                  rhs);
      },
      lhs.variant(),
      rhs.variant());
}

#define MAKE_COMPARE_LAMBDA(op) \
  [](const std::int64_t lhs, const std::int64_t rhs) { return lhs op rhs; }

bool ParseLogicalExpr(const EquationStaticLogical& expr);

bool ParseLogicalExprImpl(
    const EQ<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(expr.tuple(), MAKE_COMPARE_LAMBDA(==));
}

bool ParseLogicalExprImpl(
    const LT<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(
      expr.tuple(),
      [](const std::int64_t lhs, const std::int64_t rhs) { return lhs < rhs; });
}

bool ParseLogicalExprImpl(
    const GT<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(
      expr.tuple(),
      [](const std::int64_t lhs, const std::int64_t rhs) { return lhs > rhs; });
}

bool ParseLogicalExprImpl(
    const NE<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(expr.tuple(), MAKE_COMPARE_LAMBDA(!=));
}

bool ParseLogicalExprImpl(
    const GE<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(expr.tuple(), MAKE_COMPARE_LAMBDA(>=));
}

bool ParseLogicalExprImpl(
    const LE<EquationStaticValue, EquationStaticValue>& expr) {
  return CalculateLogicalExprImpl(expr.tuple(), MAKE_COMPARE_LAMBDA(<=));
}

bool ParseLogicalExprImpl(const And<Logical<EquationStaticValue>,
                                    Logical<EquationStaticValue>>& expr) {
  const auto& [lhs, rhs] = expr.tuple();
  return ParseLogicalExpr(rhs) && ParseLogicalExpr(rhs);
}

bool ParseLogicalExprImpl(const Or<Logical<EquationStaticValue>,
                                   Logical<EquationStaticValue>>& expr) {
  const auto& [lhs, rhs] = expr.tuple();
  return ParseLogicalExpr(rhs) || ParseLogicalExpr(rhs);
}

bool ParseLogicalExprImpl(const Not<Logical<EquationStaticValue>>& expr) {
  const auto& [unpacked_expr] = expr.tuple();
  return !ParseLogicalExpr(unpacked_expr);
}

bool ParseLogicalExpr(const EquationStaticLogical& expr) {
  return std::visit(
      [&](const auto& impl) { return ParseLogicalExprImpl(impl); },
      expr.variant());
}

void NaiveConditionalEqualHandler::Where(
    const EquationStaticLogical& logical) const {
  bool valid_logical = ParseLogicalExpr(logical);
  if (valid_logical) {
    ctx_->AddEquations(equations_);
  }
}

std::optional<std::int64_t> GetArgDimSizeImpl(
    const tIn<ArgDimPosDescriptor>& in_arg_dim_pos,
    const GetArgStaticDimT& GetInDim,
    const GetArgStaticDimT& GetOutDim) {
  return GetInDim(in_arg_dim_pos.value().tensor_idx,
                  in_arg_dim_pos.value().dim_idx);
}

std::optional<std::int64_t> GetArgDimSizeImpl(
    const tOut<ArgDimPosDescriptor>& out_arg_dim_pos,
    const GetArgStaticDimT& GetInDim,
    const GetArgStaticDimT& GetOutDim) {
  return GetOutDim(out_arg_dim_pos.value().tensor_idx,
                   out_arg_dim_pos.value().dim_idx);
}

std::optional<std::int64_t> GetArgDimSizeImpl(
    const Undefined&,
    const GetArgStaticDimT& GetInDim,
    const GetArgStaticDimT& GetOutDim) {
  LOG(FATAL) << "position not found";
}

std::optional<std::int64_t> GetArgDimSize(const OpArgDimPos& arg_dim_pos,
                                          const GetArgStaticDimT& GetInDim,
                                          const GetArgStaticDimT& GetOutDim) {
  return std::visit(
      [&](const auto& impl) {
        return GetArgDimSizeImpl(impl, GetInDim, GetOutDim);
      },
      arg_dim_pos.variant());
}

std::int64_t NaiveOpEquationContext::GetDimSize(const Dim& dim) const {
  const auto& arg_dim_pos = GetArgDimPosDescriptor(dim);
  const auto& option_dim_size =
      GetArgDimSize(arg_dim_pos, GetInDim_, GetOutDim_);
  if (!option_dim_size.has_value()) {
    LOG(FATAL) << "Dim not found";
  }
  return option_dim_size.value();
}

}  // namespace cinn::adt::config
