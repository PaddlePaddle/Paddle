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
#include <typeinfo>

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_value_match_trait.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/print_value.h"
#include "paddle/cinn/adt/simplify_value.h"

namespace cinn::adt {

template <typename T, typename ExprT>
ExprT MatchAndRewrite(const ExprT& expr, const IndexExprInferContext& ctx) {
  if (cinn::adt::Match<typename T::source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr, ctx);
  } else {
    return expr;
  }
}

struct SimplifyDot {
  using source_pattern_type = IndexDotValue<Value, List<Dim>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [iterators, dims_constants] =
        value.Get<IndexDotValue<Value, Constant>>().tuple();
    List<Constant> int64_dims{};
    for (const auto& dim_constant : *dims_constants.Get<List<Constant>>()) {
      const Constant& int64_dim = ctx.GetDimSize(dim_constant.Get<Dim>());
      if (int64_dim.Has<std::int64_t>()) {
        int64_dims->emplace_back(int64_dim);
      } else {
        return IndexDotValue<Value, Constant>{SimplifyValue(iterators, ctx),
                                              dims_constants};
      }
    }
    return IndexDotValue<Value, Constant>{SimplifyValue(iterators, ctx),
                                          int64_dims};
  }
};

struct SimplifyUnDot {
  using source_pattern_type = IndexUnDotValue<Value, List<Dim>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index, dims_constants] =
        value.Get<IndexUnDotValue<Value, Constant>>().tuple();
    List<Constant> int64_dims{};
    for (const auto& dim_constant : *dims_constants.Get<List<Constant>>()) {
      const Constant& int64_dim = ctx.GetDimSize(dim_constant.Get<Dim>());
      if (int64_dim.Has<std::int64_t>()) {
        int64_dims->emplace_back(int64_dim);
      } else {
        return IndexUnDotValue<Value, Constant>{SimplifyValue(index, ctx),
                                                dims_constants};
      }
    }
    return IndexUnDotValue<Value, Constant>{SimplifyValue(index, ctx),
                                            int64_dims};
  }
};

struct SimplifyList {
  using source_pattern_type = List<Value>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    List<Value> ret{};
    for (const auto& v : *value.Get<List<Value>>()) {
      ret->emplace_back(SimplifyValue(v, ctx));
    }
    return ret;
  }
};

struct SimplifyDotUndot {
  using source_pattern_type =
      IndexDotValue<List<ListGetItem<IndexUnDotValue<Value, List<std::int64_t>>,
                                     std::int64_t>>,
                    List<std::int64_t>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_get_item_values, dot_dims] =
        value.Get<IndexDotValue<Value, Constant>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, Constant>>().tuple();
      if (!constant_idx.Get<std::int64_t>() == i) {
        return IndexDotValue<Value, Constant>{
            SimplifyValue(list_get_item_values, ctx), dot_dims};
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return IndexDotValue<Value, Constant>{
              SimplifyValue(list_get_item_values, ctx), dot_dims};
        } else {
          // do nothing
        }
      } else {
        pre_index_undot = index_undot_value;
      }
    }
    CHECK(pre_index_undot.has_value());
    const auto& [index_value, undot_dims] =
        pre_index_undot.value().Get<IndexUnDotValue<Value, Constant>>().tuple();
    CHECK(dot_dims.Has<List<Constant>>());
    CHECK(undot_dims.Has<List<Constant>>());
    if (dot_dims == undot_dims) {
      return index_value;
    }
    return IndexDotValue<Value, Constant>{
        SimplifyValue(list_get_item_values, ctx), dot_dims};
  }
};

struct SimplifyUndotDot {
  using source_pattern_type = ListGetItem<
      IndexUnDotValue<IndexDotValue<List<Value>, List<std::int64_t>>,
                      List<std::int64_t>>,
      std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& [index_value, undot_dims] =
        index_undot_value.Get<IndexUnDotValue<Value, Constant>>().tuple();
    const auto& [index_dot_values, dot_dims] =
        index_value.Get<IndexDotValue<Value, Constant>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    CHECK(dot_dims.Has<List<Constant>>());
    CHECK(undot_dims.Has<List<Constant>>());
    if (dot_dims == undot_dims) {
      return iter_values.Get(constant_idx.Get<std::int64_t>());
    } else {
      return ListGetItem<Value, Constant>{SimplifyValue(index_undot_value, ctx),
                                          constant_idx};
    }
  }
};

struct SimplifyListGetItem {
  using source_pattern_type = ListGetItem<Value, Constant>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_values, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    return ListGetItem<Value, Constant>{SimplifyValue(list_values, ctx),
                                        constant_idx};
  }
};

struct SimplifyListGetItemList {
  using source_pattern_type = ListGetItem<List<Value>, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_values, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& iter_values = list_values.Get<List<Value>>();
    return iter_values.Get(constant_idx.Get<std::int64_t>());
  }
};

// Only simplify top-layer of value
Value SimplifyValue(Value value, const IndexExprInferContext& ctx) {
  value = MatchAndRewrite<SimplifyDot>(value, ctx);
  value = MatchAndRewrite<SimplifyUnDot>(value, ctx);
  value = MatchAndRewrite<SimplifyList>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItem>(value, ctx);
  value = MatchAndRewrite<SimplifyDotUndot>(value, ctx);
  value = MatchAndRewrite<SimplifyUndotDot>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItemList>(value, ctx);
  return value;
}

}  // namespace cinn::adt
