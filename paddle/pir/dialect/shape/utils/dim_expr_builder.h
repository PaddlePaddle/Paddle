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

#pragma once

#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace symbol {

class TEST_API DimExprBuilder {
 public:
  explicit DimExprBuilder(std::vector<DimExprConstraint>* constraints)
      : constraints_(constraints) {}

  TEST_API DimExpr ConstSize(std::int64_t dim);
  TEST_API DimExpr Symbol(const std::string& symbol_name);
  TEST_API DimExpr Add(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Any(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Mul(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Div(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Max(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Min(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API DimExpr Broadcast(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API std::vector<DimExpr> ConstShape(
      const std::vector<std::int64_t>& dims);

  TEST_API void CstrBroadcastable(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API void CstrBroadcastable(const std::vector<DimExpr>& lhs,
                                  const std::vector<DimExpr>& rhs);
  TEST_API void CstrEq(const DimExpr& lhs, const DimExpr& rhs);
  TEST_API void CstrEq(const std::vector<DimExpr>& lhs,
                       const std::vector<DimExpr>& rhs);

  TEST_API std::vector<DimExpr> Concat(const std::vector<DimExpr>& lhs,
                                       const std::vector<DimExpr>& rhs);
  TEST_API std::pair<std::vector<DimExpr>, std::vector<DimExpr>> SplitAt(
      const std::vector<DimExpr>, int index);

  TEST_API const std::vector<DimExprConstraint>& constaints() const;

 private:
  std::vector<DimExprConstraint>* constraints_;
};

}  // namespace symbol
