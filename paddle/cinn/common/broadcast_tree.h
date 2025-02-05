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
#include "paddle/common/flags.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

COMMON_DECLARE_int64(pir_broadcast_tree_limit);

namespace cinn::common {

template <typename T>
using BroadcastBranch = adt::Tuple<symbol::Broadcastable<symbol::DimExpr>,
                                   /*cstr_lhs_eq_rhs_branch*/ T,
                                   /*cstr_lhs_eq_one_branch*/ T,
                                   /*cstr_rhs_eq_one_branch*/ T>;

using BroadcastLeaf = adt::List<std::vector<symbol::DimExpr>>;

using BroadcastTree = adt::Tree<BroadcastBranch, BroadcastLeaf>;

BroadcastTree ConstructBroadcastTree(const BroadcastLeaf& leaves,
                                     int* num_of_leaves);

std::string ToTxtString(const BroadcastTree&);

std::optional<symbol::Broadcastable<symbol::DimExpr>> GetFirstCstrBroadcastable(
    const BroadcastLeaf& leaves);

}  // namespace cinn::common
