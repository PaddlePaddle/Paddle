// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/cinn/common/dim_expr_util.h"

namespace cinn::common {
using namespace symbol;  // NOLINT

namespace {

class SubstituteDimExprHelper final {
 public:
  explicit SubstituteDimExprHelper(
      const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
          pattern_to_replacement)
      : pattern_to_replacement_(pattern_to_replacement) {}

  std::optional<DimExpr> Substitute(const DimExpr& dim_expr) {
    auto iter = pattern_to_replacement_.find(dim_expr);
    if (iter != pattern_to_replacement_.end()) return iter->second;
    return std::visit([&](const auto& impl) { return SubstituteImpl(impl); },
                      dim_expr.variant());
  }

 private:
  std::optional<DimExpr> SubstituteImpl(const std::int64_t& value) {
    // `Substitute` has handled the case that `value` is matched.
    return std::nullopt;
  }
  std::optional<DimExpr> SubstituteImpl(const std::string& value) {
    // `Substitute` has handled the case that `value` is matched.
    return std::nullopt;
  }

  std::optional<DimExpr> SubstituteImpl(const Negative<DimExpr>& dim_expr) {
    return SubstituteUnary(dim_expr);
  }
  std::optional<DimExpr> SubstituteImpl(const Reciprocal<DimExpr>& dim_expr) {
    return SubstituteUnary(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteUnary(const T& dim_expr) {
    const auto& operand = dim_expr->data;
    const auto& substituted_operand = Substitute(operand);
    if (!substituted_operand.has_value()) return std::nullopt;
    return T{substituted_operand.value()};
  }

  std::optional<DimExpr> SubstituteImpl(const Add<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Mul<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Max<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Min<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  std::optional<DimExpr> SubstituteImpl(const Broadcast<DimExpr>& dim_expr) {
    return SubstituteVariadic(dim_expr);
  }

  template <typename T>
  std::optional<DimExpr> SubstituteVariadic(const T& dim_expr) {
    const auto& operands = *(dim_expr.operands);
    List<DimExpr> substituted_operands{};
    size_t replace_cnt = 0;
    for (const auto& operand : operands) {
      const auto& substituted_operand = Substitute(operand);
      replace_cnt += substituted_operand.has_value();
      substituted_operands->push_back(substituted_operand.has_value()
                                          ? substituted_operand.value()
                                          : operand);
    }
    if (replace_cnt == 0) return std::nullopt;
    return T{substituted_operands};
  }

  std::unordered_map<symbol::DimExpr, symbol::DimExpr> pattern_to_replacement_;
};

}  // namespace

symbol::DimExpr SubstituteDimExpr(
    const symbol::DimExpr& dim_expr,
    const std::unordered_map<symbol::DimExpr, symbol::DimExpr>&
        pattern_to_replacement) {
  const auto& opt_replaced =
      SubstituteDimExprHelper(pattern_to_replacement).Substitute(dim_expr);
  return opt_replaced.has_value() ? opt_replaced.value() : dim_expr;
}

}  // namespace cinn::common
