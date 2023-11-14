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
#include "paddle/cinn/adt/get_sub_reshape_dim_ranges.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/match.h"
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

struct SimplifyBroadcastedIteratorByReplacingToStaticDim {
  using source_pattern_type = BroadcastedIterator<Value, Dim>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [iterator, dim] =
        value.Get<BroadcastedIterator<Value, Constant>>().tuple();
    const Constant& int64_dim = ctx.GetDimSize(dim.Get<Dim>());
    if (int64_dim.Has<std::int64_t>()) {
      return BroadcastedIterator<Value, Constant>{
          iterator, int64_dim.Get<std::int64_t>()};
    } else {
      return value;
    }
  }
};

struct SimplifyBroadcastedIterator {
  using source_pattern_type = BroadcastedIterator<Value, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [iterator, dim] =
        value.Get<BroadcastedIterator<Value, Constant>>().tuple();
    if (dim.Get<std::int64_t>() == 1) {
      return Constant{std::int64_t(0)};
    } else {
      return iterator;
    }
  }
};

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

struct SimplifyGcdShape {
  using source_pattern_type = ListGetItem<
      IndexUnDotValue<IndexDotValue<List<Value>, List<std::int64_t>>,
                      List<std::int64_t>>,
      std::int64_t>;

  bool IsConstantListAllPositiveInt64(const List<Constant>& constants) {
    for (const auto& constant : *constants) {
      if (!constant.Has<std::int64_t>() || constant.Get<std::int64_t>() <= 0) {
        return false;
      }
    }
    return true;
  }

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
    const auto& undot_dim_values = undot_dims.Get<List<Constant>>();
    const auto& dot_dim_values = dot_dims.Get<List<Constant>>();
    CHECK(IsConstantListAllPositiveInt64(undot_dim_values));
    CHECK(IsConstantListAllPositiveInt64(dot_dim_values));

    const auto& sub_reshape_dim_ranges =
        GetSubReshapeDimRanges(undot_dim_values, dot_dim_values);
    if (!sub_reshape_dim_ranges.has_value()) {
      return ListGetItem<Value, Constant>{SimplifyValue(index_undot_value, ctx),
                                          constant_idx};
    }
    const auto& [undot_dim_ranges, dot_dim_ranges] =
        sub_reshape_dim_ranges.value();
    if (undot_dim_ranges.size() >= 1) {
      const auto& [sub_range_idx, sub_range_item_idx] = GetSubRangeItemIdx(
          undot_dim_ranges, constant_idx.Get<std::int64_t>());
      List<Constant> sub_range_undot_dims = GetSubRangeDotDims(
          undot_dim_values, undot_dim_ranges.at(sub_range_idx));
      List<Value> sub_range_dot_iterators = GetSubRangeDotIterators(
          iter_values, dot_dim_ranges.at(sub_range_idx));
      List<Constant> sub_range_dot_dims =
          GetSubRangeDotDims(dot_dim_values, dot_dim_ranges.at(sub_range_idx));
      if (sub_range_dot_dims == sub_range_undot_dims) {
        return sub_range_dot_iterators.Get(sub_range_item_idx);
      } else {
        IndexDotValue<Value, Constant> sub_range_dot{sub_range_dot_iterators,
                                                     sub_range_dot_dims};
        if (sub_range_undot_dims->size() == 1) {
          CHECK_EQ(sub_range_item_idx, 0);
          return sub_range_dot;
        } else {
          IndexUnDotValue<Value, Constant> sub_range_undot{
              sub_range_dot, sub_range_undot_dims};
          return ListGetItem<Value, Constant>{sub_range_undot,
                                              sub_range_item_idx};
        }
      }
    }
    return ListGetItem<Value, Constant>{SimplifyValue(index_undot_value, ctx),
                                        constant_idx};
  }

  std::pair<int, int> GetSubRangeItemIdx(
      const std::vector<std::pair<int, int>>& ranges,
      std::int64_t index) const {
    for (std::size_t i = 0; i < ranges.size(); ++i) {
      const auto& [begin, end] = ranges.at(i);
      if (index >= begin && index < end) {
        return std::pair<int, int>{i, index - begin};
      }
    }
  }

  List<Value> GetSubRangeDotIterators(const List<Value>& iterators,
                                      const std::pair<int, int>& range) const {
    return GetSubRange<List<Value>>(iterators, range);
  }

  List<Constant> GetSubRangeDotDims(const List<Constant>& dims,
                                    const std::pair<int, int>& range) const {
    return GetSubRange<List<Constant>>(dims, range);
  }

  template <typename ContainerT>
  ContainerT GetSubRange(const ContainerT& container,
                         const std::pair<int, int>& range) const {
    CheckRange(container, range);
    ContainerT ret{};
    ret->assign(std::next(container->begin(), range.first),
                std::next(container->begin(), range.second));
    return ret;
  }

  template <typename ContainerT>
  void CheckRange(const ContainerT& container,
                  const std::pair<int, int>& range) const {
    CHECK_GE(range.first, 0);
    CHECK_GE(range.second, 0);
    CHECK_LE(range.first, container->size());
    CHECK_LE(range.second, container->size());
    CHECK_LT(range.first, range.second);
  }
};

struct SimplifyDotDot {
  using source_pattern_type = IndexDotValue<List<Value>, List<std::int64_t>>;

  std::int64_t Product(const List<Constant>& dims) {
    std::int64_t ret = 1;
    for (const auto& dim : *dims) {
      CHECK(dim.Has<std::int64_t>());
      ret *= dim.Get<std::int64_t>();
    }
    return ret;
  }

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_dot_values, dot_dims] =
        value.Get<IndexDotValue<Value, Constant>>().tuple();
    CHECK_EQ(index_dot_values.Get<List<Value>>()->size(),
             dot_dims.Get<List<Constant>>()->size());
    List<Value> new_dot_values{};
    List<Constant> new_dot_dims{};
    for (std::size_t i = 0; i < index_dot_values.Get<List<Value>>()->size();
         ++i) {
      const auto& index_dot_value = index_dot_values.Get<List<Value>>()->at(i);
      const auto& dot_dim =
          dot_dims.Get<List<Constant>>()->at(i).Get<std::int64_t>();
      if (Match<source_pattern_type>(index_dot_value)) {
        const auto& [sub_index_dot_values, sub_dot_dims] =
            index_dot_value.Get<IndexDotValue<Value, Constant>>().tuple();
        const auto& sub_dot_dim_values = sub_dot_dims.Get<List<Constant>>();
        std::int64_t dim_product = Product(sub_dot_dim_values);
        if (dim_product == dot_dim) {
          for (std::size_t j = 0;
               j < sub_index_dot_values.Get<List<Value>>()->size();
               ++j) {
            const auto& sub_index_dot_value =
                sub_index_dot_values.Get<List<Value>>()->at(j);
            const auto& sub_dot_dim = sub_dot_dim_values->at(j);
            new_dot_values->emplace_back(sub_index_dot_value);
            new_dot_dims->emplace_back(sub_dot_dim);
          }
        } else {
          new_dot_values->emplace_back(index_dot_value);
          new_dot_dims->emplace_back(dot_dim);
        }
      } else {
        new_dot_values->emplace_back(index_dot_value);
        new_dot_dims->emplace_back(dot_dim);
      }
    }
    return IndexDotValue<Value, Constant>{new_dot_values, new_dot_dims};
  }
};

// Only simplify top-layer of value
Value SimplifyValue(Value value, const IndexExprInferContext& ctx) {
  value = MatchAndRewrite<SimplifyBroadcastedIteratorByReplacingToStaticDim>(
      value, ctx);
  value = MatchAndRewrite<SimplifyBroadcastedIterator>(value, ctx);
  value = MatchAndRewrite<SimplifyDot>(value, ctx);
  value = MatchAndRewrite<SimplifyUnDot>(value, ctx);
  value = MatchAndRewrite<SimplifyList>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItem>(value, ctx);
  value = MatchAndRewrite<SimplifyDotUndot>(value, ctx);
  value = MatchAndRewrite<SimplifyUndotDot>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItemList>(value, ctx);
  value = MatchAndRewrite<SimplifyGcdShape>(value, ctx);
  value = MatchAndRewrite<SimplifyDotDot>(value, ctx);
  return value;
}

}  // namespace cinn::adt
