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

#include "paddle/pir/include/dialect/shape/utils/constraints_manager.h"

namespace symbol {
void ConstraintsManager::AddEqCstr(const DimExpr& lhs, const DimExpr& rhs) {
  if (lhs == rhs) {
    return;
  }
  equals_.Union(lhs, rhs);
  VLOG(8) << "AddEqCstr the constraint: " << lhs << " == " << rhs;
}

void ConstraintsManager::AddBroadcastableCstr(const DimExpr& lhs,
                                              const DimExpr& rhs) {
  broadcastables_.push_back(Broadcastable<DimExpr>(lhs, rhs));
}

void ConstraintsManager::AddGTOneCstr(const DimExpr& dim_expr) {
  gtones_.insert(dim_expr);
}

bool ConstraintsManager::IsDimExprEqual(const DimExpr& lhs,
                                        const DimExpr& rhs) {
  if (equals_.Find(lhs) == equals_.Find(rhs)) {
    return true;
  }
  return false;
}

void ConstraintsManager::PrintDimExprClusters() {
  std::vector<std::vector<DimExpr>> equal_clusters = equals_.Clusters();
  for (auto equal_cluster : equal_clusters) {
    VLOG(0) << "Equal Clusters:";
    for (auto dim_expr : equal_cluster) {
      VLOG(0) << dim_expr;
    }
  }
}

}  // namespace symbol
