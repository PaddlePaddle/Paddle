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

#include "paddle/cinn/adt/index_expr_infer_context.h"

#include "paddle/cinn/adt/print.h"
#include "paddle/cinn/adt/union_find.h"

namespace cinn::adt {
namespace {
bool IsZeroValue(const Value& value) {
  return value.Has<DimExpr>() && value.Get<DimExpr>().Has<std::int64_t>() &&
         value.Get<DimExpr>().Get<std::int64_t>() == 0;
}
}  // namespace

void IndexExprInferContext::MatchZeroValueAndThenSet(const Variable& variable,
                                                     const Value& value) {
  if (!IsZeroValue(value)) {
    return;
  }
  CHECK(variable.Has<Iterator>());
  UnionFind<Iterator> uf;
  for (const auto& [tmp_variable, tmp_value] : variable2value_) {
    if (tmp_variable.Has<Iterator>() && tmp_value.Has<Iterator>()) {
      uf.Union(tmp_variable.Get<Iterator>(), tmp_value.Get<Iterator>());
    } else {
      // Do nothing
    }
  }
  const auto& zero_iterators = uf.NodeCluster(variable.Get<Iterator>());
  for (const auto& iterator : zero_iterators) {
    Variable zero_variable{iterator};
    variable2value_.erase(zero_variable);
    CHECK(variable2value_.emplace(zero_variable, DimExpr{0}).second);
  }
}

bool IndexExprInferContext::SetValue(const Variable& variable,
                                     const Value& value) {
  if (value.Has<Undefined>()) {
    return false;
  }
  MatchZeroValueAndThenSet(variable, value);
  if (HasValue(variable)) {
    const auto& old_value = GetValue(variable);
    CHECK(old_value == value) << "old_value: " << ToTxtString(old_value)
                              << ", new_value: " << ToTxtString(value);
  } else {
    CHECK(variable2value_.emplace(variable, value).second);
  }
  return true;
}

bool IndexExprInferContext::DimsEqual(const List<DimExpr>& lhs,
                                      const List<DimExpr>& rhs) const {
  return lhs == rhs;
}

}  // namespace cinn::adt
