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

#include "paddle/cinn/adt/equation.h"
#include "paddle/cinn/adt/equation_value.h"
#include "paddle/cinn/adt/m_expr.h"

namespace cinn::adt::equation {

class IndexExprInferContext final {
 public:
  IndexExprInferContext(const IndexExprInferContext&) = delete;
  IndexExprInferContext(IndexExprInferContext&&) = delete;

  explicit IndexExprInferContext(
      const std::unordered_map<const Variable, Value>& init_map)
      : map_(init_map) {}

  const Value& GetValue(const Variable& variable) const {
    return map_.at(variable);
  }

  auto SetValue(const Variable& variable, const Value& value) {
    return map_.emplace(variable, value);
  }

  bool HasValue(const Variable& variable) const {
    return map_.count(variable) > 0;
  }

 private:
  std::unordered_map<const Variable, Value> map_;
};

}  // namespace cinn::adt::equation
