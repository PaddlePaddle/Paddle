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

#include <unordered_map>

#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/map_expr.h"

namespace cinn::adt {

class IndexExprInferContext final {
 public:
  IndexExprInferContext(const IndexExprInferContext&) = delete;
  IndexExprInferContext(IndexExprInferContext&&) = delete;

  explicit IndexExprInferContext(
      const std::unordered_map<Variable, const Value>& init_variable2value)
      : variable2value_(init_variable2value) {}

  const Value& GetValue(const Variable& variable) const {
    return variable2value_.at(variable);
  }

  auto SetValue(const Variable& variable, const Value& value) {
    return variable2value_.emplace(variable, value);
  }

  bool HasValue(const Variable& variable) const {
    return variable2value_.count(variable) > 0;
  }

  bool DimsEqual(const List<DimExpr>& lhs, const List<DimExpr>& rhs) const;

  bool ProductEqual(const List<DimExpr>& lhs, const DimExpr& rhs) const {
    ADT_TODO();
  }

 private:
  std::unordered_map<Variable, const Value> variable2value_;
};

}  // namespace cinn::adt
