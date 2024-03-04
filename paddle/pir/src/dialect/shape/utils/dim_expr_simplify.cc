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

#include "paddle/pir/include/dialect/shape/utils/dim_expr_simplify.h"
#include <numeric>

namespace symbol {

namespace {

template <typename T>
DimExpr TrySimplifyPass(const DimExpr& expr) {
  if (!expr.Has<typename T::dim_expr_type>()) {
    return expr;
  }
  return T().Rewrite(expr);
}

DimExpr Simplify(const DimExpr& expr);

/*
 * Simplify Example:
 * Negative(dim_expr) => Negative(Simplify(dim_expr))
 */
template <template <typename> class Op>
struct SimplifyOneOperand {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    auto [operand] = *expr.Get<Op<DimExpr>>();
    const auto& ret_operand = Simplify(operand);
    if (ret_operand == operand) {
      return expr;
    } else {
      return Op<DimExpr>{ret_operand};
    }
    LOG(FATAL) << "Dead code.";
  }
};

template <template <typename> class Op>
struct SimplifyOneOperandTrait;

/*
 * Simplify Example:
 * Negative(DimExpr(0)) => DimExpr(0)
 * Reciprocal(DimExpr(1)) => DimExpr(1)
 */
template <template <typename> class Op>
struct SimplifyUnitOneOperand {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operand] = *expr.Get<Op<DimExpr>>();
    if (operand.template Has<std::int64_t>() &&
        operand.template Get<std::int64_t>() ==
            SimplifyOneOperandTrait<Op>::unit) {
      return DimExpr{SimplifyOneOperandTrait<Op>::unit};
    } else {
      return expr;
    }
    LOG(FATAL) << "Dead code";
  }
};

template <>
struct SimplifyOneOperandTrait<Negative> {
  static constexpr std::int64_t unit = 0;
};

template <>
struct SimplifyOneOperandTrait<Reciprocal> {
  static constexpr std::int64_t unit = 1;
};

/*
 * Simplify Example:
 * Add(dim_expr0, dim_expr1, ...) =>
 * Add(Simplify(dim_expr0), Simplify(dim_expr1), ...)
 */
template <template <typename> class Op>
struct SimplifyOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    List<DimExpr> mut_operands{};
    for (const auto& operand : *operands) {
      mut_operands->emplace_back(Simplify(operand));
    }
    if (mut_operands == operands) {
      return expr;
    } else {
      return Op<DimExpr>{mut_operands};
    }
    LOG(FATAL) << "Dead code.";
  }
};

template <typename T>
struct GetOrderValue;

template <>
struct GetOrderValue<Broadcast<DimExpr>> {
  static constexpr int value = 10;
};

template <>
struct GetOrderValue<Mul<DimExpr>> {
  static constexpr int value = 20;
};

template <>
struct GetOrderValue<Add<DimExpr>> {
  static constexpr int value = 30;
};

template <>
struct GetOrderValue<std::string> {
  static constexpr int value = 40;
};

template <>
struct GetOrderValue<std::int64_t> {
  static constexpr int value = 50;
};

template <>
struct GetOrderValue<Reciprocal<DimExpr>> {
  static constexpr int value = 60;
};

template <>
struct GetOrderValue<Negative<DimExpr>> {
  static constexpr int value = 70;
};

template <>
struct GetOrderValue<Max<DimExpr>> {
  static constexpr int value = 80;
};

template <>
struct GetOrderValue<Min<DimExpr>> {
  static constexpr int value = 90;
};

bool IsLhsBeforeRhs(const DimExpr& lhs, const DimExpr& rhs);

template <template <typename> class Op>
struct IsListLhsBeforeListRhsStruct {
  static bool Call(const Op<DimExpr>& lhs, const Op<DimExpr>& rhs) {
    const auto& [lhs_operands] = lhs;
    const auto& [rhs_operands] = rhs;
    if (lhs_operands->size() < rhs_operands->size()) {
      return true;
    }
    if (lhs_operands->size() > rhs_operands->size()) {
      return false;
    }
    for (std::size_t i = 0; i < lhs_operands->size(); ++i) {
      if (!IsLhsBeforeRhs(lhs_operands->at(i), rhs_operands->at(i))) {
        return false;
      }
    }
    return true;
  }
};

template <typename T0, typename T1>
struct IsLhsBeforeRhsStruct {
  static bool Call(const T0& lhs, const T1& rhs) {
    return GetOrderValue<T0>::value < GetOrderValue<T1>::value;
  }
};

template <>
struct IsLhsBeforeRhsStruct<std::int64_t, std::int64_t> {
  static bool Call(std::int64_t lhs, std::int64_t rhs) { return lhs < rhs; }
};

template <>
struct IsLhsBeforeRhsStruct<std::string, std::string> {
  static bool Call(const std::string& lhs, const std::string& rhs) {
    return lhs.compare(rhs) < 0;
  }
};

template <>
struct IsLhsBeforeRhsStruct<Negative<DimExpr>, Negative<DimExpr>> {
  static bool Call(const Negative<DimExpr>& lhs, const Negative<DimExpr>& rhs) {
    const auto& lhs_operand = lhs->data;
    const auto& rhs_operand = rhs->data;
    return IsLhsBeforeRhs(lhs_operand, rhs_operand);
  }
};

template <>
struct IsLhsBeforeRhsStruct<Reciprocal<DimExpr>, Reciprocal<DimExpr>> {
  static bool Call(const Reciprocal<DimExpr>& lhs,
                   const Reciprocal<DimExpr>& rhs) {
    const auto& lhs_operand = lhs->data;
    const auto& rhs_operand = rhs->data;
    return IsLhsBeforeRhs(lhs_operand, rhs_operand);
  }
};

template <>
struct IsLhsBeforeRhsStruct<Add<DimExpr>, Add<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Add> {};

template <>
struct IsLhsBeforeRhsStruct<Mul<DimExpr>, Mul<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Mul> {};

template <>
struct IsLhsBeforeRhsStruct<Broadcast<DimExpr>, Broadcast<DimExpr>> final
    : public IsListLhsBeforeListRhsStruct<Broadcast> {};

bool IsLhsBeforeRhs(const DimExpr& lhs, const DimExpr& rhs) {
  return std::visit(
      [&](const auto& lhs, const auto& rhs) {
        return IsLhsBeforeRhsStruct<std::decay_t<decltype(lhs)>,
                                    std::decay_t<decltype(rhs)>>::Call(lhs,
                                                                       rhs);
      },
      lhs.variant(),
      rhs.variant());
}

/*
 * Sort operands in DimExpr
 */
template <template <typename> class Op>
struct SortOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    if (operands->size() == 1) {
      return operands->at(0);
    }
    bool is_sorted = IsSorted(operands);
    if (is_sorted) {
      return expr;
    }
    List<DimExpr> mut_operands{};
    mut_operands->insert(
        mut_operands->end(), operands->begin(), operands->end());
    std::sort(mut_operands->begin(), mut_operands->end(), &IsLhsBeforeRhs);
    return Op<DimExpr>{mut_operands};
  }

  bool IsSorted(const List<DimExpr>& operands) {
    CHECK(!operands->empty());
    for (std::size_t i = 0; i < operands->size() - 1; ++i) {
      if (IsLhsBeforeRhs(operands->at(i + 1), operands->at(i))) {
        return false;
      }
    }
    return true;
  }
};

std::int64_t GetInteger(const DimExpr& expr) {
  if (expr.Has<Negative<DimExpr>>()) {
    const auto& integer = expr.Get<Negative<DimExpr>>()->data;
    CHECK(integer.Has<std::int64_t>());
    return -integer.Get<std::int64_t>();
  }
  CHECK(expr.Has<std::int64_t>());
  return expr.Get<std::int64_t>();
}

template <template <typename> class Op, template <typename> class Inversed>
struct VisitEachInversableOperandStruct {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    if (expr.Has<Op<DimExpr>>()) {
      const auto& [operands] = expr.Get<Op<DimExpr>>();
      for (const auto& operand : *operands) {
        Call(operand, DoEach, depth + 1, is_inversed);
      }
    } else if (expr.Has<Inversed<DimExpr>>()) {
      const auto& [operand] = expr.Get<Inversed<DimExpr>>();
      Call(operand->data, DoEach, depth, !is_inversed);
    } else {
      DoEach(expr, depth, is_inversed);
    }
  }
};

template <template <typename> class>
struct VisitEachOperandStruct;

template <>
struct VisitEachOperandStruct<Add>
    : public VisitEachInversableOperandStruct<Add, Negative> {};

template <>
struct VisitEachOperandStruct<Mul>
    : public VisitEachInversableOperandStruct<Mul, Reciprocal> {};

template <>
struct VisitEachOperandStruct<Broadcast> {
  template <typename DoEachT>
  static void Call(const DimExpr& expr,
                   const DoEachT& DoEach,
                   std::size_t depth,
                   bool is_inversed) {
    if (expr.Has<Broadcast<DimExpr>>()) {
      const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
      for (const auto& operand : *operands) {
        Call(operand, DoEach, depth + 1, false);
      }
    } else {
      DoEach(expr, depth, false);
    }
  }
};

template <template <typename> class Op, typename DoEachT>
void VisitEachOperand(const DimExpr& expr, const DoEachT& DoEach) {
  if (expr.Has<Op<DimExpr>>()) {
    VisitEachOperandStruct<Op>::Call(
        expr, DoEach, /*depth=*/0, /*is_inversed=*/false);
  } else {
    // Do nothing;
  }
}

template <template <typename> class Op>
bool HasNested(const DimExpr& expr) {
  bool has_nested = false;
  VisitEachOperand<Op>(
      expr, [&](const DimExpr& operand, std::size_t depth, bool is_negative) {
        has_nested = has_nested || (depth > 0);
      });
  return has_nested;
}

template <template <typename> class Op>
struct GetInversed {};

template <>
struct GetInversed<Add> {
  static DimExpr Call(const DimExpr& expr) { return Negative<DimExpr>(expr); }
};

template <>
struct GetInversed<Mul> {
  static DimExpr Call(const DimExpr& expr) { return Reciprocal<DimExpr>(expr); }
};

template <>
struct GetInversed<Broadcast> {
  static DimExpr Call(const DimExpr& expr) {
    LOG(FATAL) << "Broadcast is not a group in math.";
  }
};

/*
 * Simplify Example:
 * Add(dim_expr0, Add(dim_expr1, dim_expr2)) =>
 * Add(dim_expr0, dim_expr1, dim_expr2)
 */
template <template <typename> class Op>
struct FlattenOperands {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    if (!HasNested<Op>(expr)) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    VisitEachOperand<Op>(
        expr,
        [&](const DimExpr& dim_expr, std::size_t depth, bool is_negative) {
          if (is_negative) {
            ret_operands->emplace_back(GetInversed<Op>::Call(dim_expr));
          } else {
            ret_operands->emplace_back(dim_expr);
          }
        });
    return Op<DimExpr>{ret_operands};
  }
};

template <template <typename> class Op>
struct FoldOperandTrait;

template <template <typename> class Op>
size_t GetConstDimCount(const List<DimExpr>& exprs) {
  std::size_t cnt = 0;
  for (const auto& expr : *exprs) {
    cnt += FoldOperandTrait<Op>::IsConstPattern(expr);
  }
  return cnt;
}

/*
 * Simplify Example:
 * Add(dim_expr0, 0) => dim_expr0
 */
template <template <typename> class Op>
struct FoldUnitConstant {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto [operands] = expr.Get<Op<DimExpr>>();
    if (GetConstDimCount<Op>(operands) == 0) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    for (const auto& operand : *operands) {
      if (FoldOperandTrait<Op>::IsUnitDimExpr(operand)) {
        continue;
      } else {
        ret_operands->emplace_back(operand);
      }
    }
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    LOG(FATAL) << "Dead code.";
  }
};

/*
 * Simplify Example:
 * Add(dim_expr0, 0, 1, 2) => Add(dim_expr0, 3)
 */
template <template <typename> class Op>
struct FoldConstants {
  using dim_expr_type = Op<DimExpr>;

  DimExpr Rewrite(const DimExpr& expr) {
    const auto [operands] = expr.Get<Op<DimExpr>>();
    if (GetConstDimCount<Op>(operands) <= 1) {
      return expr;
    }
    List<DimExpr> ret_operands{};
    typename FoldOperandTrait<Op>::const_value_type const_dim =
        FoldOperandTrait<Op>::MakeUnit();
    for (const auto& operand : *operands) {
      if (FoldOperandTrait<Op>::IsConstPattern(operand)) {
        FoldOperandTrait<Op>::Accumulate(&const_dim, operand);
      } else {
        ret_operands->emplace_back(operand);
      }
    }
    if (!FoldOperandTrait<Op>::IsUnit(const_dim)) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(const_dim, &ret_operands);
    }
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    LOG(FATAL) << "Dead code.";
  }
};

template <>
struct FoldOperandTrait<Add> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Negative<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Negative<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return 0; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = *value + GetInteger(expr);
  }
  static bool IsUnit(const const_value_type& value) { return value == 0; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 0;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }

  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    if (lhs.Has<Negative<DimExpr>>()) {
      const auto& lhs_operand = lhs.Get<Negative<DimExpr>>()->data;
      return lhs_operand == rhs;
    }
    if (rhs.Has<Negative<DimExpr>>()) {
      const auto& [rhs_operand] = *rhs.Get<Negative<DimExpr>>();
      return rhs_operand == lhs;
    }
    return false;
  }
};

using ConstRational = std::pair<std::int64_t, std::int64_t>;

ConstRational SimplifiedConstRational(int64_t num, int64_t dem) {
  std::int64_t gcd = std::gcd(num, dem);
  return ConstRational{num / gcd, dem / gcd};
}

template <typename T>
std::optional<ConstRational> GetConstRationalImpl(const T& expr) {
  LOG(FATAL) << "not supported.";
  return std::nullopt;
}

std::optional<ConstRational> GetConstRationalImpl(std::int64_t value) {
  return ConstRational{value, 1};
}

std::optional<ConstRational> GetConstRationalImpl(
    const Reciprocal<DimExpr>& value) {
  const auto& [denominator] = *value;
  return ConstRational{1, denominator.Get<std::int64_t>()};
}

ConstRational GetConstRational(const DimExpr& expr) {
  return std::visit(
      [&](const auto& impl) {
        std::optional<ConstRational> opt_ret = GetConstRationalImpl(impl);
        CHECK(opt_ret.has_value());
        return opt_ret.value();
      },
      expr.variant());
}

ConstRational MulConstRational(const ConstRational& lhs,
                               const ConstRational& rhs) {
  const auto [lhs_num, lhs_dem] = lhs;
  const auto [rhs_num, rhs_dem] = rhs;
  // Crossing is correct.
  const auto [simplifed_lhs_num, simplifed_rhs_dem] =
      SimplifiedConstRational(lhs_num, rhs_dem);
  const auto [simplifed_rhs_num, simplifed_lhs_dem] =
      SimplifiedConstRational(rhs_num, lhs_dem);
  return ConstRational{simplifed_lhs_num * simplifed_rhs_num,
                       simplifed_lhs_dem * simplifed_rhs_dem};
}

template <>
struct FoldOperandTrait<Mul> {
  using const_value_type = ConstRational;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    if (dim_expr.Has<Reciprocal<DimExpr>>()) {
      const auto& operand = dim_expr.Get<Reciprocal<DimExpr>>()->data;
      return operand.Has<std::int64_t>();
    }
    return false;
  }

  static const_value_type MakeUnit() { return ConstRational{1, 1}; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    *value = MulConstRational(*value, GetConstRational(expr));
  }
  static bool IsUnit(const const_value_type& value) {
    return value.first == 1 && value.second == 1;
  }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    const auto& [num, dem] = value;
    (*ret)->emplace_back(num);
    CHECK_NE(dem, 0);
    if (dem != 1) {
      (*ret)->emplace_back(Reciprocal<DimExpr>{DimExpr{dem}});
    }
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    if (lhs.Has<Reciprocal<DimExpr>>()) {
      const auto& lhs_operand = lhs.Get<Reciprocal<DimExpr>>()->data;
      return lhs_operand == rhs;
    }
    if (rhs.Has<Reciprocal<DimExpr>>()) {
      const auto& rhs_operand = rhs.Get<Reciprocal<DimExpr>>()->data;
      return rhs_operand == lhs;
    }
    return false;
  }
};

template <>
struct FoldOperandTrait<Broadcast> {
  using const_value_type = std::int64_t;

  static bool IsConstPattern(const DimExpr& dim_expr) {
    if (dim_expr.Has<std::int64_t>()) {
      return true;
    }
    return false;
  }

  static const_value_type MakeUnit() { return 1; }
  static void Accumulate(const_value_type* value, const DimExpr& expr) {
    CHECK(expr.Has<std::int64_t>());
    std::int64_t expr_value = expr.Get<std::int64_t>();
    if (*value == 1) {
      *value = expr_value;
    } else if (expr_value != 1) {
      CHECK_EQ(*value, expr_value);
    } else {
      // do nothing.
    }
  }
  static bool IsUnit(const const_value_type& value) { return value == 1; }
  static bool IsUnitDimExpr(const DimExpr& dim_expr) {
    if (!dim_expr.Has<std::int64_t>()) {
      return false;
    }
    return dim_expr.Get<std::int64_t>() == 1;
  }
  static void MakeAndAppendDimExpr(const const_value_type& value,
                                   List<DimExpr>* ret) {
    (*ret)->emplace_back(value);
  }
  static bool IsInversedPair(const DimExpr& lhs, const DimExpr& rhs) {
    return false;
  }
};

/*
 * Simplify Example:
 * Mul(2, Reciprocal(2)) => 1
 */
template <template <typename> class Op>
struct FoldInversedPairToUnit {
  using dim_expr_type = Op<DimExpr>;

  struct SearchResult {
    std::size_t value_pos;
    std::size_t inverse_value_pos;
  };

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Op<DimExpr>>();
    const auto& opt_searched = SearchInversedPair(operands);
    if (!opt_searched.has_value()) {
      return expr;
    }
    const auto& [i, j] = opt_searched.value();
    List<DimExpr> ret_operands{};
    ret_operands->insert(ret_operands->end(),
                         operands->begin(),
                         std::next(operands->begin(), i));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), i + 1),
                         std::next(operands->begin(), j));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), j + 1),
                         operands->end());
    if (ret_operands->empty()) {
      FoldOperandTrait<Op>::MakeAndAppendDimExpr(
          FoldOperandTrait<Op>::MakeUnit(), &ret_operands);
    }
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Op<DimExpr>{ret_operands};
    }
    LOG(FATAL) << "Dead code";
  }

  std::optional<SearchResult> SearchInversedPair(
      const List<DimExpr>& operands) {
    for (std::size_t i = 0; i < operands->size(); ++i) {
      for (std::size_t j = 0; j < operands->size(); ++j) {
        if (i == j) {
          continue;
        }
        if (FoldOperandTrait<Op>::IsInversedPair(operands->at(i),
                                                 operands->at(j))) {
          return SearchResult{i, j};
        }
      }
    }
    return std::nullopt;
  }
};

/*
 * Simplify Example:
 * Broadcast(2, dim_expr) => 2
 */
struct FoldRedundantSymbolicBroadcast {
  using dim_expr_type = Broadcast<DimExpr>;

  struct MaxInt64 {
    std::int64_t value;
    int value_pos;
  };

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
    const auto& opt_max_int64 = SearchMaxInt64(operands);
    if (!opt_max_int64.has_value()) {
      return expr;
    }
    const auto& [value, i] = opt_max_int64.value();
    if (value != 1) {
      return value;
    }
    List<DimExpr> ret_operands{};
    ret_operands->insert(ret_operands->end(),
                         operands->begin(),
                         std::next(operands->begin(), i));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), i + 1),
                         operands->end());
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Broadcast<DimExpr>{ret_operands};
    }
    LOG(FATAL) << "Dead code.";
  }

  std::optional<MaxInt64> SearchMaxInt64(const List<DimExpr>& operands) {
    std::optional<MaxInt64> ret;
    int operands_size = static_cast<int>(operands->size());
    for (int i = 0; i < operands_size; ++i) {
      const auto& expr = operands->at(i);
      if (!expr.Has<std::int64_t>()) {
        continue;
      }
      std::int64_t int64_value = expr.Get<std::int64_t>();
      if (ret.has_value()) {
        if (int64_value > 1) {
          if (ret.value().value > 1) {
            CHECK_EQ(ret.value().value, int64_value);
          }
          ret = MaxInt64{int64_value, i};
        }
      } else {
        ret = MaxInt64{int64_value, i};
      }
    }
    return ret;
  }
};

/*
 * Simplify Example:
 * Broadcast(dim_expr0, Broadcast(dim_expr1, dim_expr2)) =>
 * Broadcast(dim_expr0, dim_expr1, dim_expr2)
 */
struct FoldRedundantBroadcast {
  using dim_expr_type = Broadcast<DimExpr>;

  struct SearchResult {
    std::size_t value_pos;
    std::size_t same_value_pos;
  };

  DimExpr Rewrite(const DimExpr& expr) {
    const auto& [operands] = expr.Get<Broadcast<DimExpr>>();
    const auto& opt_searched = SearchInversedPair(operands);
    if (!opt_searched.has_value()) {
      return expr;
    }
    const auto& [i, _] = opt_searched.value();
    List<DimExpr> ret_operands{};
    ret_operands->insert(ret_operands->end(),
                         operands->begin(),
                         std::next(operands->begin(), i));
    ret_operands->insert(ret_operands->end(),
                         std::next(operands->begin(), i + 1),
                         operands->end());
    if (ret_operands->size() == 1) {
      return ret_operands->at(0);
    } else {
      return Broadcast<DimExpr>{ret_operands};
    }
    LOG(FATAL) << "Dead code.";
  }

  std::optional<SearchResult> SearchInversedPair(
      const List<DimExpr>& operands) {
    for (std::size_t i = 0; i < operands->size(); ++i) {
      for (std::size_t j = 0; j < operands->size(); ++j) {
        if (i == j) {
          continue;
        }
        if (operands->at(i) == operands->at(j)) {
          return SearchResult{i, j};
        }
      }
    }
    return std::nullopt;
  }
};
template <typename PassT>
void DoPass(bool* rewrited, DimExpr* expr) {
  const auto old_expr = *expr;
  *expr = TrySimplifyPass<PassT>(*expr);
  *rewrited = *rewrited || (old_expr != *expr);
}

DimExpr Simplify(const DimExpr& expr) {
  DimExpr ret = expr;
  for (bool keep_rewrite = true; keep_rewrite;) {
    keep_rewrite = false;
    DoPass<SimplifyOneOperand<Negative>>(&keep_rewrite, &ret);
    DoPass<SimplifyOneOperand<Reciprocal>>(&keep_rewrite, &ret);
    DoPass<SimplifyUnitOneOperand<Negative>>(&keep_rewrite, &ret);
    DoPass<SimplifyUnitOneOperand<Reciprocal>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Add>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<SimplifyOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Add>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<SortOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Add>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Mul>>(&keep_rewrite, &ret);
    DoPass<FlattenOperands<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Add>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Mul>>(&keep_rewrite, &ret);
    DoPass<FoldUnitConstant<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Add>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Mul>>(&keep_rewrite, &ret);
    DoPass<FoldConstants<Broadcast>>(&keep_rewrite, &ret);
    DoPass<FoldInversedPairToUnit<Add>>(&keep_rewrite, &ret);
    DoPass<FoldInversedPairToUnit<Mul>>(&keep_rewrite, &ret);
    DoPass<FoldRedundantBroadcast>(&keep_rewrite, &ret);
    DoPass<FoldRedundantSymbolicBroadcast>(&keep_rewrite, &ret);
  }
  return ret;
}

}  // namespace

DimExpr SimplifyDimExpr(const DimExpr& expr) { return Simplify(expr); }

}  // namespace symbol
