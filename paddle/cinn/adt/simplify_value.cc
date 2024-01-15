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

#include "paddle/cinn/adt/dim_expr_simplifier.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_value_match_trait.h"
#include "paddle/cinn/adt/get_sub_reshape_dim_ranges.h"
#include "paddle/cinn/adt/index_expr_infer_context.h"
#include "paddle/cinn/adt/match.h"
#include "paddle/cinn/adt/simplify_value.h"

namespace cinn::adt {

namespace {

template <typename T, typename ExprT>
ExprT MatchAndRewrite(const ExprT& expr, const IndexExprInferContext& ctx) {
  if (cinn::adt::Match<typename T::source_pattern_type>(expr)) {
    return T().MatchAndRewrite(expr, ctx);
  } else {
    return expr;
  }
}

struct SimplifyBroadcastedIterator {
  using source_pattern_type = BroadcastedIterator<Value, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [iterator, dim] =
        value.Get<BroadcastedIterator<Value, DimExpr>>().tuple();
    if (dim.Get<std::int64_t>() == 1) {
      return DimExpr{std::int64_t(0)};
    } else {
      return iterator;
    }
  }
};

struct SimplifyRedundantBroadcastedIterator {
  using source_pattern_type =
      BroadcastedIterator<BroadcastedIterator<Value, DimExpr>, DimExpr>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [outter_iterator, outter_dim] =
        value.Get<BroadcastedIterator<Value, DimExpr>>().tuple();
    const auto& [inner_iterator, inner_dim] =
        outter_iterator.Get<BroadcastedIterator<Value, DimExpr>>().tuple();

    if (outter_dim == inner_dim) {
      return SimplifyValue(outter_iterator, ctx);
    } else {
      const auto& bd = MakeBroadcastedDim(outter_dim, inner_dim);
      const auto& simplified_bd = DimExpr{SimplifyDimExpr(bd)};
      return BroadcastedIterator<Value, DimExpr>{inner_iterator, simplified_bd};
    }
    LOG(FATAL) << "Dead code";
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
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, DimExpr>>().tuple();
      if (constant_idx.Get<std::int64_t>() != i) {
        return IndexDotValue<Value, List<DimExpr>>{
            SimplifyValue(list_get_item_values, ctx), dot_dims};
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return IndexDotValue<Value, List<DimExpr>>{
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
        pre_index_undot.value()
            .Get<IndexUnDotValue<Value, List<DimExpr>>>()
            .tuple();
    if (dot_dims == undot_dims) {
      return index_value;
    }
    return IndexDotValue<Value, List<DimExpr>>{
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
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& [index_value, undot_dims] =
        index_undot_value.Get<IndexUnDotValue<Value, List<DimExpr>>>().tuple();
    const auto& [index_dot_values, dot_dims] =
        index_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    if (dot_dims == undot_dims) {
      return iter_values.Get(constant_idx.Get<std::int64_t>());
    } else {
      return ListGetItem<Value, DimExpr>{SimplifyValue(index_undot_value, ctx),
                                         constant_idx};
    }
  }
};

struct SimplifyListGetItem {
  using source_pattern_type = ListGetItem<Value, DimExpr>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_values, constant_idx] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    return ListGetItem<Value, DimExpr>{SimplifyValue(list_values, ctx),
                                       constant_idx};
  }
};

// Match: IndexDotValue([i_59, i_60], [sym_17, 1])
// After: IndexDotValue([i_59], [sym_17])
struct SimplifyIndexDotValue {
  using source_pattern_type = IndexDotValue<List<Value>, List<DimExpr>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_dot_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iterators = index_dot_values.Get<List<Value>>();
    List<Value> ret_iterators{};
    List<DimExpr> ret_dot_dims{};
    for (std::int64_t i = 0; i < dot_dims->size(); ++i) {
      const auto& dot_dim = dot_dims->at(i);
      if (dot_dim.Has<std::int64_t>() && dot_dim.Get<std::int64_t>() == 1) {
        // Do nothing
      } else {
        ret_iterators->emplace_back(iterators->at(i));
        ret_dot_dims->emplace_back(dot_dim);
      }
    }

    if (ret_iterators != iterators || ret_dot_dims != dot_dims) {
      CHECK_EQ(ret_iterators->size(), ret_dot_dims->size());
      CHECK(!ret_iterators->empty());
      return IndexDotValue<Value, List<DimExpr>>{ret_iterators, ret_dot_dims};
    } else {
      return value;
    }
    LOG(FATAL) << "Dead code";
  }
};

// Match: IndexDotValue([BI(i_59, sym_17), i_60], [sym_17, 64])
// After: IndexDotValue([i_59, i_60], [sym_17, 64])
struct SymbolicDim_SimplifyDotBroadcatedIterator {
  using source_pattern_type = IndexDotValue<List<Value>, List<DimExpr>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_dot_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iterators = index_dot_values.Get<List<Value>>();
    List<Value> ret_iterators{};
    List<DimExpr> ret_dot_dims{};

    const auto& GetSimplifiedIteratorAndDim =
        [](const BroadcastedIterator<Value, DimExpr>& value,
           const SymbolicDim& sym_dim) -> std::pair<Value, DimExpr> {
      const auto& [broadcast_iterator, dim_expr] = value.tuple();
      if (broadcast_iterator.Has<Iterator>() && dim_expr.Has<SymbolicDim>()) {
        CHECK(dim_expr.Get<SymbolicDim>() == sym_dim);
        return std::make_pair(broadcast_iterator, dim_expr);
      } else {
        return std::make_pair(value, sym_dim);
      }
      LOG(FATAL) << "Dead code";
    };

    for (std::int64_t i = 0; i < dot_dims->size(); ++i) {
      const auto& dot_dim = dot_dims->at(i);
      const auto& iterator = iterators->at(i);
      if (iterator.Has<BroadcastedIterator<Value, DimExpr>>() &&
          dot_dim.Has<SymbolicDim>()) {
        const auto& [tmp_iterator, tmp_dim] = GetSimplifiedIteratorAndDim(
            iterator.Get<BroadcastedIterator<Value, DimExpr>>(),
            dot_dim.Get<SymbolicDim>());
        ret_iterators->emplace_back(tmp_iterator);
        ret_dot_dims->emplace_back(tmp_dim);
      } else {
        ret_iterators->emplace_back(iterators->at(i));
        ret_dot_dims->emplace_back(dot_dim);
      }
    }

    if (ret_iterators != iterators || ret_dot_dims != dot_dims) {
      CHECK_EQ(ret_iterators->size(), ret_dot_dims->size());
      CHECK(!ret_iterators->empty());
      return IndexDotValue<Value, List<DimExpr>>{ret_iterators, ret_dot_dims};
    } else {
      return value;
    }
    LOG(FATAL) << "Dead code";
  }
};

// Match: ListGetItem(IndexUnDot(ok, [sym_17, 1]), 1)
// After: 0
// Match: ListGetItem(IndexUnDot(ok, [sym_17, 1, 64]), 2)
// After: ListGetItem(IndexUnDot(ok, [sym_17, 64]), 1)
struct SimplifyListOfIndexUnDotValue {
  using source_pattern_type =
      ListGetItem<IndexUnDotValue<Value, List<DimExpr>>, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_get_item_value, constant] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    std::int64_t list_idx = constant.Get<std::int64_t>();
    std::int64_t undot_dim_before_get_item_cnt = 0;
    const auto& [index_undot_value, undot_dims] =
        list_get_item_value.Get<IndexUnDotValue<Value, List<DimExpr>>>()
            .tuple();
    List<DimExpr> ret_undot_dims{};
    for (std::int64_t i = 0; i < undot_dims->size(); ++i) {
      const auto& undot_dim = undot_dims->at(i);
      if (undot_dim.Has<std::int64_t>() && undot_dim.Get<std::int64_t>() == 1) {
        if (i < list_idx) {
          undot_dim_before_get_item_cnt++;
        } else if (i == list_idx) {
          return Value{DimExpr{0}};
        } else {
          // Do nothing
        }
      } else {
        ret_undot_dims->emplace_back(undot_dim);
      }
    }
    if (ret_undot_dims != undot_dims) {
      const auto& ret_undot_value = IndexUnDotValue<Value, List<DimExpr>>{
          index_undot_value, ret_undot_dims};
      return ListGetItem<Value, DimExpr>{
          ret_undot_value, DimExpr{list_idx - undot_dim_before_get_item_cnt}};
    } else {
      return value;
    }
    LOG(FATAL) << "Dead code";
  }
};

struct SimplifyListGetItemList {
  using source_pattern_type = ListGetItem<List<Value>, std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_values, constant_idx] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& iter_values = list_values.Get<List<Value>>();
    return iter_values.Get(constant_idx.Get<std::int64_t>());
  }
};

struct SimplifyGcdShape {
  using source_pattern_type = ListGetItem<
      IndexUnDotValue<IndexDotValue<List<Value>, List<std::int64_t>>,
                      List<std::int64_t>>,
      std::int64_t>;

  bool IsConstantListAllPositiveInt64(const List<DimExpr>& constants) {
    for (const auto& constant : *constants) {
      if (!constant.Has<std::int64_t>() || constant.Get<std::int64_t>() <= 0) {
        return false;
      }
    }
    return true;
  }

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& [index_value, undot_dims] =
        index_undot_value.Get<IndexUnDotValue<Value, List<DimExpr>>>().tuple();
    const auto& [index_dot_values, dot_dims] =
        index_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    const auto& undot_dim_values = undot_dims;
    const auto& dot_dim_values = dot_dims;
    CHECK(IsConstantListAllPositiveInt64(undot_dim_values));
    CHECK(IsConstantListAllPositiveInt64(dot_dim_values));

    const auto& sub_reshape_dim_ranges =
        GetSubReshapeDimRanges(undot_dim_values, dot_dim_values);
    if (!sub_reshape_dim_ranges.has_value()) {
      return ListGetItem<Value, DimExpr>{SimplifyValue(index_undot_value, ctx),
                                         constant_idx};
    }
    const auto& [undot_dim_ranges, dot_dim_ranges] =
        sub_reshape_dim_ranges.value();
    if (undot_dim_ranges.size() >= 1) {
      const auto& [sub_range_idx, sub_range_item_idx] = GetSubRangeItemIdx(
          undot_dim_ranges, constant_idx.Get<std::int64_t>());
      List<DimExpr> sub_range_undot_dims = GetSubRangeDotDims(
          undot_dim_values, undot_dim_ranges.at(sub_range_idx));
      List<Value> sub_range_dot_iterators = GetSubRangeDotIterators(
          iter_values, dot_dim_ranges.at(sub_range_idx));
      List<DimExpr> sub_range_dot_dims =
          GetSubRangeDotDims(dot_dim_values, dot_dim_ranges.at(sub_range_idx));
      if (sub_range_dot_dims == sub_range_undot_dims) {
        return sub_range_dot_iterators.Get(sub_range_item_idx);
      } else {
        IndexDotValue<Value, List<DimExpr>> sub_range_dot{
            sub_range_dot_iterators, sub_range_dot_dims};
        if (sub_range_undot_dims->size() == 1) {
          CHECK_EQ(sub_range_item_idx, 0);
          return sub_range_dot;
        } else {
          IndexUnDotValue<Value, List<DimExpr>> sub_range_undot{
              sub_range_dot, sub_range_undot_dims};
          return ListGetItem<Value, DimExpr>{sub_range_undot,
                                             sub_range_item_idx};
        }
      }
    }
    return ListGetItem<Value, DimExpr>{SimplifyValue(index_undot_value, ctx),
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

  List<DimExpr> GetSubRangeDotDims(const List<DimExpr>& dims,
                                   const std::pair<int, int>& range) const {
    return GetSubRange<List<DimExpr>>(dims, range);
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

  std::int64_t Product(const List<DimExpr>& dims) {
    std::int64_t ret = 1;
    for (const auto& dim : *dims) {
      CHECK(dim.Has<std::int64_t>());
      ret *= dim.Get<std::int64_t>();
    }
    return ret;
  }

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_dot_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    CHECK_EQ(index_dot_values.Get<List<Value>>()->size(), dot_dims->size());
    List<Value> new_dot_values{};
    List<DimExpr> new_dot_dims{};
    for (std::size_t i = 0; i < index_dot_values.Get<List<Value>>()->size();
         ++i) {
      const auto& index_dot_value = index_dot_values.Get<List<Value>>()->at(i);
      const auto& dot_dim = dot_dims->at(i).Get<std::int64_t>();
      if (Match<source_pattern_type>(index_dot_value)) {
        const auto& [sub_index_dot_values, sub_dot_dims] =
            index_dot_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
        const auto& sub_dot_dim_values = sub_dot_dims;
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
    return IndexDotValue<Value, List<DimExpr>>{new_dot_values, new_dot_dims};
  }
};

struct SymbolicDim_SimplifyDotUndot {
  using source_pattern_type = IndexDotValue<
      List<ListGetItem<IndexUnDotValue<Value, List<DimExpr>>, std::int64_t>>,
      List<DimExpr>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_get_item_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, DimExpr>>().tuple();
      if (constant_idx.Get<std::int64_t>() != i) {
        return IndexDotValue<Value, List<DimExpr>>{
            SimplifyValue(list_get_item_values, ctx), dot_dims};
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return IndexDotValue<Value, List<DimExpr>>{
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
        pre_index_undot.value()
            .Get<IndexUnDotValue<Value, List<DimExpr>>>()
            .tuple();
    const auto& dot_dim_values = dot_dims;
    const auto& undot_dim_values = undot_dims;
    if (ctx.DimsEqual(dot_dim_values, undot_dim_values)) {
      return index_value;
    } else {
      return IndexDotValue<Value, List<DimExpr>>{
          SimplifyValue(list_get_item_values, ctx), dot_dims};
    }
    LOG(FATAL) << "Dead code";
  }
};

struct SymbolicDim_SimplifyDotUndot_DimExpr {
  using source_pattern_type = IndexDotValue<
      List<ListGetItem<
          IndexUnDotValue<Value, List<Union<DimExpr, std::int64_t>>>,
          std::int64_t>>,
      List<Union<DimExpr, std::int64_t>>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [list_get_item_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& list_get_items = list_get_item_values.Get<List<Value>>();
    std::optional<Value> pre_index_undot{std::nullopt};
    for (std::size_t i = 0; i < list_get_items->size(); ++i) {
      const auto& [index_undot_value, constant_idx] =
          list_get_items.Get(i).Get<ListGetItem<Value, DimExpr>>().tuple();
      if (constant_idx.Get<std::int64_t>() != i) {
        return IndexDotValue<Value, List<DimExpr>>{
            SimplifyValue(list_get_item_values, ctx), dot_dims};
      }
      if (pre_index_undot.has_value()) {
        if (!(pre_index_undot.value() == index_undot_value)) {
          return IndexDotValue<Value, List<DimExpr>>{
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
        pre_index_undot.value()
            .Get<IndexUnDotValue<Value, List<DimExpr>>>()
            .tuple();
    const auto& dot_dim_values = dot_dims;
    const auto& undot_dim_values = undot_dims;
    if (dot_dim_values == undot_dim_values) {
      return index_value;
    } else {
      return IndexDotValue<Value, List<DimExpr>>{
          SimplifyValue(list_get_item_values, ctx), dot_dims};
    }
    LOG(FATAL) << "Dead code";
  }
};

struct SymbolicDim_SimplifyUndotDot {
  using source_pattern_type = ListGetItem<
      IndexUnDotValue<IndexDotValue<List<Value>, List<DimExpr>>, List<DimExpr>>,
      std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& [index_value, undot_dims] =
        index_undot_value.Get<IndexUnDotValue<Value, List<DimExpr>>>().tuple();
    const auto& [index_dot_values, dot_dims] =
        index_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    if (ctx.DimsEqual(dot_dims, undot_dims)) {
      return iter_values.Get(constant_idx.Get<std::int64_t>());
    } else {
      return ListGetItem<Value, DimExpr>{SimplifyValue(index_undot_value, ctx),
                                         constant_idx};
    }
  }
};

struct SymbolicDim_SimplifyUndotDot_DimExpr {
  using source_pattern_type = ListGetItem<
      IndexUnDotValue<
          IndexDotValue<List<Value>, List<Union<DimExpr, std::int64_t>>>,
          List<Union<DimExpr, std::int64_t>>>,
      std::int64_t>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_undot_value, constant_idx] =
        value.Get<ListGetItem<Value, DimExpr>>().tuple();
    const auto& [index_value, undot_dims] =
        index_undot_value.Get<IndexUnDotValue<Value, List<DimExpr>>>().tuple();
    const auto& [index_dot_values, dot_dims] =
        index_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    const auto& iter_values = index_dot_values.Get<List<Value>>();
    if (dot_dims == undot_dims) {
      return iter_values.Get(constant_idx.Get<std::int64_t>());
    } else {
      return ListGetItem<Value, DimExpr>{SimplifyValue(index_undot_value, ctx),
                                         constant_idx};
    }
  }
};

struct SymbolicDim_SimplifyDotDot {
  using source_pattern_type = IndexDotValue<List<Value>, List<DimExpr>>;

  Value MatchAndRewrite(const Value& value, const IndexExprInferContext& ctx) {
    const auto& [index_dot_values, dot_dims] =
        value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
    CHECK_EQ(index_dot_values.Get<List<Value>>()->size(), dot_dims->size());
    List<Value> new_dot_values{};
    List<DimExpr> new_dot_dims{};
    for (std::size_t i = 0; i < index_dot_values.Get<List<Value>>()->size();
         ++i) {
      const auto& index_dot_value = index_dot_values.Get<List<Value>>()->at(i);
      DimExpr dot_dim = dot_dims->at(i);
      if (Match<source_pattern_type>(index_dot_value)) {
        const auto& [sub_index_dot_values, sub_dot_dims] =
            index_dot_value.Get<IndexDotValue<Value, List<DimExpr>>>().tuple();
        const auto& sub_dot_dim_values = sub_dot_dims;
        if (ctx.ProductEqual(sub_dot_dim_values, dot_dim)) {
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
    return IndexDotValue<Value, List<DimExpr>>{new_dot_values, new_dot_dims};
  }
};

}  // namespace

// Only simplify top-layer of value
Value SimplifyValue(Value value, const IndexExprInferContext& ctx) {
  value = MatchAndRewrite<SimplifyList>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItem>(value, ctx);
  value = MatchAndRewrite<SimplifyIndexDotValue>(value, ctx);
  value = MatchAndRewrite<SimplifyListOfIndexUnDotValue>(value, ctx);
  value = MatchAndRewrite<SimplifyBroadcastedIterator>(value, ctx);
  value = MatchAndRewrite<SimplifyRedundantBroadcastedIterator>(value, ctx);
  value = MatchAndRewrite<SimplifyDotUndot>(value, ctx);
  value = MatchAndRewrite<SimplifyUndotDot>(value, ctx);
  value = MatchAndRewrite<SimplifyListGetItemList>(value, ctx);
  value = MatchAndRewrite<SimplifyGcdShape>(value, ctx);
  value = MatchAndRewrite<SimplifyDotDot>(value, ctx);
  // For symbolic dim simplification
  // value =
  // MatchAndRewrite<SymbolicDim_SimplifyDotBroadcatedIterator>(value, ctx);
  value = MatchAndRewrite<SymbolicDim_SimplifyDotUndot>(value, ctx);
  value = MatchAndRewrite<SymbolicDim_SimplifyUndotDot>(value, ctx);
  // value = MatchAndRewrite<SymbolicDim_SimplifyGcdShape>(value, ctx);
  value = MatchAndRewrite<SymbolicDim_SimplifyDotDot>(value, ctx);
  // For DimExpr
  value = MatchAndRewrite<SymbolicDim_SimplifyDotUndot_DimExpr>(value, ctx);
  value = MatchAndRewrite<SymbolicDim_SimplifyUndotDot_DimExpr>(value, ctx);

  return value;
}

}  // namespace cinn::adt
