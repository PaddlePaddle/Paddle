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
  using source_pattern_type = IndexDot<Value, List<Stride>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [iterators, strides_constants] =
        value.Get<IndexDot<Value, Constant>>().tuple();
    List<Constant> int64_strides{};
    for (const auto& stride_constant :
         *strides_constants.Get<List<Constant>>()) {
      const Constant& int64_stride =
          ctx.GetStrideSize(stride_constant.Get<Stride>());
      if (int64_stride.Has<std::int64_t>()) {
        int64_strides->emplace_back(int64_stride);
      } else {
        return IndexDot<Value, Constant>{SimplifyValue(iterators, ctx),
                                         strides_constants};
      }
    }
    return IndexDot<Value, Constant>{SimplifyValue(iterators, ctx),
                                     int64_strides};
  }
};

struct SimplifyUnDot {
  using source_pattern_type = IndexUnDot<Value, List<Stride>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index, strides_constants] =
        value.Get<IndexUnDot<Value, Constant>>().tuple();
    List<Constant> int64_strides{};
    for (const auto& stride_constant :
         *strides_constants.Get<List<Constant>>()) {
      const Constant& int64_stride =
          ctx.GetStrideSize(stride_constant.Get<Stride>());
      if (int64_stride.Has<std::int64_t>()) {
        int64_strides->emplace_back(int64_stride);
      } else {
        return IndexUnDot<Value, Constant>{SimplifyValue(index, ctx),
                                           strides_constants};
      }
    }
    return IndexUnDot<Value, Constant>{SimplifyValue(index, ctx),
                                       int64_strides};
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
  using source_pattern_type = IndexDot<
      List<ListGetItem<IndexUnDot<Value, List<std::int64_t>>, std::int64_t>>,
      List<std::int64_t>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_get_item_values, dot_strides] =
        value.Get<IndexDot<Value, Constant>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, Constant>>().tuple();
      if (!constant_idx.Get<std::int64_t>() == i) {
        return IndexDot<Value, Constant>{
            SimplifyValue(list_get_item_values, ctx), dot_strides};
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return IndexDot<Value, Constant>{
              SimplifyValue(list_get_item_values, ctx), dot_strides};
        } else {
          // do nothing
        }
      } else {
        pre_index_undot = index_undot_value;
      }
    }
    CHECK(pre_index_undot.has_value());
    const auto& [index_value, undot_strides] =
        pre_index_undot.value().Get<IndexUnDot<Value, Constant>>().tuple();
    CHECK(dot_strides.Has<List<Constant>>());
    CHECK(undot_strides.Has<List<Constant>>());
    if (dot_strides == undot_strides) {
      return index_value;
    }
    return IndexDot<Value, Constant>{SimplifyValue(list_get_item_values, ctx),
                                     dot_strides};
  }
};

struct SimplifyUndotDot {
  using source_pattern_type = ListGetItem<
      IndexUnDot<IndexDot<List<Value>, List<std::int64_t>>, List<std::int64_t>>,
      std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, Constant>>().tuple();
    const auto& [index_value, undot_strides] =
        index_undot_value.Get<IndexUnDot<Value, Constant>>().tuple();
    const auto& [index_dot_values, dot_strides] =
        index_value.Get<IndexDot<Value, Constant>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    CHECK(dot_strides.Has<List<Constant>>());
    CHECK(undot_strides.Has<List<Constant>>());
    if (dot_strides == undot_strides) {
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
