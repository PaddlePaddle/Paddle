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

#include <optional>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/match_trait.h"
#include "paddle/cinn/adt/simplify_value.h"

namespace cinn::adt::equation {

template <typename T, typename ExprT>
ExprT MatchAndRewrite(const ExprT& expr) {
  if (cinn::adt::Match<typename T::source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr);
  } else {
    return expr;
  }
}

struct SimplifyDotUndot {
  using source_pattern_type =
      IndexDot<List<ListGetItem<IndexUnDot<Value>, std::int64_t>>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [dot_strides, list_get_item_values] =
        value.Get<IndexDot<Value>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, Constant>>().tuple();
      if (!constant_idx.Get<std::int64_t>() == i) {
        return value;
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return value;
        } else {
          // do nothing
        }
      } else {
        pre_index_undot = index_undot_value;
      }
    }
    CHECK(pre_index_undot.has_value());
    const auto& [undot_strides, index_value] =
        pre_index_undot.value().Get<IndexUnDot<Value>>().tuple();
    if (dot_strides != undot_strides) {
      return value;
    }
    return index_value;
  }
};

struct SimplifyUndotDot {
  using source_pattern_type =
      ListGetItem<IndexUnDot<IndexDot<List<Value>>>, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& [undot_strides, index_value] =
        index_undot_value.Get<IndexUnDot<Value>>().tuple();
    const auto& [dot_strides, index_dot_values] =
        index_value.Get<IndexDot<Value>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    if (undot_strides != dot_strides) {
      return value;
    }
    return iter_values.Get(constant_idx.Get<std::int64_t>());
  }
};

struct SimplifyListGetItem {
  using source_pattern_type = ListGetItem<List<Value>, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_values, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& iter_values = list_values.Get<List<Value>>();
    return iter_values.Get(constant_idx.Get<std::int64_t>());
  }
};

// Only simplify top-layer of value
Value SimplifyValue(const IndexExprInferContext& ctx, Value value) {
  value = MatchAndRewrite<SimplifyDotUndot>(value, ctx);
  value = MatchAndRewrite<SimplifyUndotDot>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItem>(value, ctx);
  return value;
}

}  // namespace cinn::adt::equation
