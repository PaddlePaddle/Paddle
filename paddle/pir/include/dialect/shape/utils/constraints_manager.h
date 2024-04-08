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

#pragma once

#include <unordered_set>

#include "paddle/common/union_find_set.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace symbol {

class IR_API ConstraintsManager {
 public:
  void AddEqCstr(const DimExpr& lhs, const DimExpr& rhs);

  bool IsEqual(const DimExpr& lhs, const DimExpr& rhs);

  void AddGTOneCstr(const DimExpr& dim_expr);

  bool IsGTOne(const DimExpr& dim_expr);

  void AddBCableCstr(const DimExpr& lhs, const DimExpr& rhs);

  void PrintDimExprClusters(std::stringstream& ss);

  using EqualCallbackFunc = std::function<void(const DimExpr&, const DimExpr&)>;
  void SetEqualCallbackFunc(EqualCallbackFunc equal_callback_func);

 private:
  void SubstituteInConstraint(const DimExpr& lhs, const DimExpr& rhs);

  std::pair<DimExpr, DimExpr> SimplifyEqualCstr(const DimExpr& lhs,
                                                const DimExpr& rhs);

 private:
  EqualCallbackFunc equal_callback_func_ = nullptr;

  using EqualConstraints = common::UnionFindSet<DimExpr>;
  using GTOneConstraints = std::unordered_set<DimExpr>;
  using BCableConstraints = std::vector<Broadcastable<DimExpr>>;

  EqualConstraints equals_;
  GTOneConstraints gtones_;
  BCableConstraints bcables_;
};

}  // namespace symbol
