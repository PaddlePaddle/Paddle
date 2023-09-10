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

#include <unordered_map>
#include <variant>

#include "glog/logging.h"

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_solver.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/simplify_value.h"
#include "paddle/cinn/adt/tags.h"

namespace cinn::adt::equation::value {

class IndexExprInferContext;

std::unordered_map<Variable, Value> InferValues(
    const Identity<tOut<Iterator>, tIn<Iterator>>& id,
    equation::IndexExprInferContext* ctx) {
  const auto& [out_iter, in_iter] = id.tuple();
  Variable in_variable{in_iter.value()};
  CHECK(ctx->HasValue(in_variable));
  return {{out_iter.value(), ctx->GetValue(in_variable)}};
}

std::unordered_map<Variable, Value> InferValues(
    const Identity<tOut<Index>, tIn<Index>>& id,
    equation::IndexExprInferContext* ctx) {
  const auto& [out_index, in_index] = id.tuple();
  Variable in_variable{in_index.value()};
  CHECK(ctx->HasValue(in_variable));
  return {{out_index.value(), ctx->GetValue(in_variable)}};
}

std::unordered_map<Variable, Value> InferValues(
    const Dot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot,
    equation::IndexExprInferContext* ctx) {
  const auto& [strides, out_index, in_iters] = dot.tuple();
  List<Value> in_values;
  for (const auto& iter : *in_iters.value()) {
    in_values->emplace_back(ctx->GetValue(iter));
  }
  List<Constant> stride_constants{strides->begin(), strides->end()};
  IndexDot<Value> index_dot{stride_constants, in_values};
  return {{out_index.value(), index_dot}};
}

std::unordered_map<Variable, Value> InferValues(
    const UnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot,
    equation::IndexExprInferContext* ctx) {
  const auto& [strides, out_iters, in_index] = undot.tuple();
  List<Constant> stride_constants{strides->begin(), strides->end()};
  IndexUnDot<Value> index_undot{stride_constants,
                                ctx->GetValue(in_index.value())};

  std::unordered_map<Variable, Value> ret{};
  for (std::size_t idx = 0; idx < out_iters.value()->size(); ++idx) {
    ListGetItem<Value, Constant> list_get_item{index_undot, idx};
    ret.emplace(out_iters.value()->at(idx), list_get_item);
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValues(
    const InMsgBox2OutMsgBox<tOut<tOutMsgBox<OpArgIndexes>>,
                             tIn<tInMsgBox<OpArgIndexes>>>&
        in_msg_box2out_msg_box,
    equation::IndexExprInferContext* ctx) {
  const auto& [op_placeholder, out_box_indexes, in_box_indexes] =
      in_msg_box2out_msg_box.tuple();
  const auto& [out_box_in_indexes, out_box_out_indexes] =
      out_box_indexes.value().value().tuple();
  const auto& [in_box_in_indexes, in_box_out_indexes] =
      in_box_indexes.value().value().tuple();
  std::unordered_map<Variable, Value> ret{{op_placeholder, adt::Ok{}}};
  CHECK_EQ(out_box_in_indexes->size(), in_box_in_indexes->size());
  CHECK_EQ(out_box_out_indexes->size(), in_box_out_indexes->size());
  for (std::size_t i = 0; i < out_box_in_indexes->size(); ++i) {
    const auto& value = ctx->GetValue(in_box_in_indexes->at(i));
    CHECK(ret.emplace(out_box_in_indexes->at(i), value).second);
  }
  for (std::size_t i = 0; i < out_box_out_indexes->size(); ++i) {
    const auto& value = ctx->GetValue(in_box_in_indexes->at(i));
    CHECK(ret.emplace(out_box_out_indexes->at(i), value).second);
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValues(
    const Function* function, equation::IndexExprInferContext* ctx) {
  return std::visit([&](auto&& function) { return InferValues(function, ctx); },
                    function->variant());
}

DEFINE_ADT_TAG(tHasUniqueInferedValue);

tHasUniqueInferedValue<bool> MergeInferedValuesIntoCtx(
    const Function* function, equation::IndexExprInferContext* ctx) {
  auto output_variable2value = InferValues(function, ctx);
  for (const auto& [variable, unsimplified_value] : output_variable2value) {
    Value simplified_value({SimplifyValue(*ctx, unsimplified_value)});
    if (!ctx->HasValue(variable)) {
      ctx->SetValue(variable, simplified_value);
    } else {
      const Value& old_value = ctx->GetValue(variable);
      if (simplified_value != old_value) {
        return tHasUniqueInferedValue<bool>{false};
      }
    }
  }
  return tHasUniqueInferedValue<bool>{true};
}

void SolveEquations(
    const EquationGraphTopoWalker<const Variable, const Function*>& walker,
    const std::vector<Variable>& starts,
    equation::IndexExprInferContext* ctx) {
  walker.WalkFunction(
      starts.begin(), starts.end(), [&](const Function* function) {
        tHasUniqueInferedValue<bool> has_unique_value =
            MergeInferedValuesIntoCtx(function, ctx);
        CHECK(has_unique_value.value());
      });
}

bool IsEquationsSolvable(
    const EquationGraphTopoWalker<const Variable, const Function*>& walker,
    const Variable& start,
    equation::IndexExprInferContext* ctx) {
  bool is_solvable = true;

  const auto& HasConflictInferedValue = [&](const Function* function) {
    tHasUniqueInferedValue<bool> has_unique_value =
        MergeInferedValuesIntoCtx(function, ctx);
    return !has_unique_value.value();
  };

  walker.WalkFunction(start, [&](const Function* function) {
    if (is_solvable && HasConflictInferedValue(function)) {
      is_solvable = false;
    }
  });
  return is_solvable;
}

}  // namespace cinn::adt::equation::value
