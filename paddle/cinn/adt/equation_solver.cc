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
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/print_value.h"
#include "paddle/cinn/adt/simplify_value.h"
#include "paddle/cinn/adt/tags.h"
#include "paddle/cinn/common/equation_graph_topo_walker.h"

namespace cinn::adt {

std::unordered_map<Variable, Value> InferValuesImpl(
    const Identity<tOut<Iterator>, tIn<Iterator>>& id,
    IndexExprInferContext* ctx) {
  const auto& [out_iter, in_iter] = id.tuple();
  Variable in_variable{in_iter.value()};
  CHECK(ctx->HasValue(in_variable));
  return {{out_iter.value(), ctx->GetValue(in_variable)}};
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const Identity<tOut<Index>, tIn<Index>>& id, IndexExprInferContext* ctx) {
  const auto& [out_index, in_index] = id.tuple();
  Variable in_variable{in_index.value()};
  CHECK(ctx->HasValue(in_variable));
  return {{out_index.value(), ctx->GetValue(in_variable)}};
}

bool HasReplicatedValues(const List<Value>& values) {
  for (std::size_t i = 0; i < values->size(); ++i) {
    for (std::size_t j = i + 1; j < values->size(); ++j) {
      if (values->at(i) == values->at(j)) {
        return true;
      }
    }
  }
  return false;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const IndexDot<List<Stride>, tOut<Index>, tIn<List<Iterator>>>& dot,
    IndexExprInferContext* ctx) {
  const auto& [strides, out_index, in_iters] = dot.tuple();
  List<Value> in_values;
  for (const auto& iter : *in_iters.value()) {
    in_values->emplace_back(ctx->GetValue(iter));
  }
  if (HasReplicatedValues(in_values)) {
    return {{out_index.value(), Undefined{}}};
  }
  List<Constant> stride_constants{};
  for (const auto& stride : *strides) {
    stride_constants->emplace_back(stride);
  }
  IndexDotValue<Value, Constant> index_dot{in_values, stride_constants};
  return {{out_index.value(), index_dot}};
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const IndexUnDot<List<Stride>, tOut<List<Iterator>>, tIn<Index>>& undot,
    IndexExprInferContext* ctx) {
  const auto& [strides, out_iters, in_index] = undot.tuple();

  List<Constant> stride_constants{};
  for (const auto& stride : *strides) {
    stride_constants->emplace_back(stride);
  }
  IndexUnDotValue<Value, Constant> index_undot{ctx->GetValue(in_index.value()),
                                               stride_constants};

  std::unordered_map<Variable, Value> ret{};
  for (std::size_t idx = 0; idx < out_iters.value()->size(); ++idx) {
    ListGetItem<Value, Constant> list_get_item{index_undot, idx};
    ret.emplace(out_iters.value()->at(idx), list_get_item);
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const InMsg2OutMsg<tOut<FakeOpPlaceHolder>,
                       tOut<OpArgIndexes<std::optional<Index>>>,
                       tIn<OpArgIndexes<Index>>>& in_msg2out_msg,
    IndexExprInferContext* ctx) {
  const auto& [op_placeholder, out_msg_indexes, in_msg_indexes] =
      in_msg2out_msg.tuple();
  const auto& [out_msg_in_indexes, out_msg_out_indexes] =
      out_msg_indexes.value().tuple();
  const auto& [in_msg_in_indexes, in_msg_out_indexes] =
      in_msg_indexes.value().tuple();
  std::unordered_map<Variable, Value> ret{{op_placeholder.value(), Ok{}}};
  CHECK_EQ(out_msg_in_indexes.value()->size(),
           in_msg_in_indexes.value()->size());
  CHECK_EQ(out_msg_out_indexes.value()->size(),
           in_msg_out_indexes.value()->size());
  for (std::size_t i = 0; i < out_msg_in_indexes.value()->size(); ++i) {
    const auto& value = ctx->GetValue(in_msg_in_indexes.value()->at(i));
    CHECK(ret.emplace(out_msg_in_indexes.value()->at(i), value).second);
  }
  for (std::size_t i = 0; i < out_msg_out_indexes.value()->size(); ++i) {
    const auto& value = ctx->GetValue(in_msg_out_indexes.value()->at(i));
    const auto& out_index = out_msg_out_indexes.value()->at(i);
    if (out_index.has_value()) {
      CHECK(ret.emplace(out_index.value(), value).second);
    }
  }
  return ret;
}

std::unordered_map<Variable, Value> InferValuesImpl(
    const ConstantFunction<tOut<Iterator>, tIn<Index>>& constant_function,
    IndexExprInferContext* ctx) {
  const auto& [out_iter, in_index, constant] = constant_function.tuple();
  return std::unordered_map<Variable, Value>{{out_iter.value(), constant}};
}

std::unordered_map<Variable, Value> InferValues(const Function* function,
                                                IndexExprInferContext* ctx) {
  return std::visit(
      [&](auto&& function) { return InferValuesImpl(function, ctx); },
      function->variant());
}

DEFINE_ADT_TAG(tValueInferSuccess);

template <typename OnFailT>
tValueInferSuccess<bool> MergeInferedValuesIntoCtx(const Function* function,
                                                   IndexExprInferContext* ctx,
                                                   const OnFailT& OnFail) {
  auto output_variable2value = InferValues(function, ctx);
  for (const auto& [variable, unsimplified_value] : output_variable2value) {
    Value simplified_value({SimplifyValue(unsimplified_value, *ctx)});
    if (simplified_value.Has<Undefined>()) {
      return OnFail(std::optional<Value>{std::nullopt}, simplified_value);
    }
    if (!ctx->HasValue(variable)) {
      ctx->SetValue(variable, simplified_value);
    } else {
      std::optional<Value> opt_old_value = ctx->GetValue(variable);
      if (simplified_value != opt_old_value.value()) {
        return OnFail(opt_old_value, simplified_value);
      }
    }
  }
  return tValueInferSuccess<bool>{true};
}

tValueInferSuccess<bool> MergeInferedValuesIntoCtx(const Function* function,
                                                   IndexExprInferContext* ctx) {
  return MergeInferedValuesIntoCtx(
      function, ctx, [&](const auto&, const auto&) {
        return tValueInferSuccess<bool>{false};
      });
}

void SolveEquations(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const std::vector<Variable>& starts,
    IndexExprInferContext* ctx) {
  walker.WalkFunction(
      starts.begin(), starts.end(), [&](const Function* function) {
        tValueInferSuccess<bool> has_unique_value =
            MergeInferedValuesIntoCtx(function, ctx);
        CHECK(has_unique_value.value());
      });
}

std::string GetDebugString(const std::optional<Value>& opt_old_value) {
  if (opt_old_value.has_value()) {
    return DebugString(opt_old_value.value());
  } else {
    return "";
  }
}

void CheckEquationsSolvable(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const Variable& start,
    IndexExprInferContext* ctx) {
  const auto& CheckNoConflictInferedValue = [&](const Function* function) {
    MergeInferedValuesIntoCtx(
        function,
        ctx,
        [&](const auto& opt_old_value, const auto& simplified_value) {
          LOG(ERROR) << "old_value: " << GetDebugString(opt_old_value);
          LOG(ERROR) << "simplified_value: " << DebugString(simplified_value);
          LOG(FATAL) << "CheckEquationsSolvable Failed";
          return tValueInferSuccess<bool>{false};
        });
  };

  walker.WalkFunction(start, CheckNoConflictInferedValue);
}

tHasNoConflictValue<bool> TrySolveEquations(
    const EquationGraphTopoWalker<Variable, const Function*>& walker,
    const Variable& start,
    IndexExprInferContext* ctx) {
  bool has_no_conflict_value = true;

  const auto& HasConflictInferedValue = [&](const Function* function) {
    tValueInferSuccess<bool> has_unique_value =
        MergeInferedValuesIntoCtx(function, ctx);
    return !has_unique_value.value();
  };

  walker.WalkFunction(start, [&](const Function* function) {
    if (has_no_conflict_value && HasConflictInferedValue(function)) {
      has_no_conflict_value = false;
    }
  });
  return tHasNoConflictValue<bool>{has_no_conflict_value};
}

}  // namespace cinn::adt
