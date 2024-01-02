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

#include "paddle/cinn/adt/tree.h"
#include "paddle/pir/dialect/shape/utils/dim_expr.h"

namespace cinn::common {

template <typename T>
struct BroadcastBranch {
  symbol::Broadcastable<symbol::DimExpr> broadcastable_condition;
  T cstr_lhs_eq_rhs_branch;
  T cstr_lhs_eq_one_branch;
  T cstr_rhs_eq_one_branch;
};

using BroadcastLeaf = adt::List<std::vector<symbol::DimExpr>>;

using BroadcastTree =
    adt::Tree<std::shared_ptr<BroadcastBranch>, BroadcastLeaf>;

std::shared_ptr<BroadcastTree> ConstructBroadcastTree(
    const BroadcastLeaf& leaves);

}  // namespace cinn::common
