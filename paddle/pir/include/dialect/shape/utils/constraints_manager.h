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
  ConstraintsManager() = default;
  ConstraintsManager(const ConstraintsManager&) = delete;
  ConstraintsManager(ConstraintsManager&&) = delete;

  void AddEqCstr(const DimExpr& lhs, const DimExpr& rhs);

  bool IsEqual(const DimExpr& lhs, const DimExpr& rhs) const;

  void AddGTOneCstr(const DimExpr& dim_expr);

  bool IsGTOne(const DimExpr& dim_expr) const;

  void AddBroadcastableCstr(const DimExpr& lhs, const DimExpr& rhs);

  bool IsBroadcastable(const DimExpr& lhs, const DimExpr& rhs) const;

  template <typename DoEachClusterT>
  void VisitEqualClusters(const DoEachClusterT& DoEachCluster) const;

  template <typename DoEachT>
  void EqualConstraintsVisitor(const DoEachT& DoEach);

  template <typename DoEachT>
  void GTOneConstraintsVisitor(const DoEachT& DoEach);

  template <typename DoEachT>
  void GTOneConstraintsVisitor(const DoEachT& DoEach) const;

  template <typename DoEachT>
  void BroadcastableConstraintsVisitor(const DoEachT& DoEach);

  template <typename DoEachT>
  void BroadcastableConstraintsVisitor(const DoEachT& DoEach) const;

  using EqualCallbackFunc = std::function<void(const DimExpr&, const DimExpr&)>;
  void SetEqualCallbackFunc(EqualCallbackFunc equal_callback_func);

  using EqualConstraints = common::UnionFindSet<DimExpr>;
  using GTOneConstraints = std::unordered_set<DimExpr>;
  using BroadcastableConstraints = std::unordered_set<Broadcastable<DimExpr>>;

  const EqualConstraints& equals() const { return equals_; }

  const GTOneConstraints& gtones() const { return gtones_; }

  const BroadcastableConstraints& broadcastables() const {
    return broadcastables_;
  }

 private:
  void SubstituteInConstraint(const DimExpr& lhs, const DimExpr& rhs);

 private:
  EqualCallbackFunc equal_callback_func_ = nullptr;

  EqualConstraints equals_;
  GTOneConstraints gtones_;
  BroadcastableConstraints broadcastables_;
};

std::ostream& operator<<(std::ostream& os,
                         const ConstraintsManager& constraints_manager);

}  // namespace symbol
