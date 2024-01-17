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
#include "glog/logging.h"

#include "paddle/cinn/adt/adt.h"
#include "paddle/cinn/adt/symbolic_dim.h"
#include "paddle/cinn/hlir/framework/pir/group.h"

namespace cinn::adt::adapter {

struct DynamicTensor final {
  ::pir::Value node_data;
  const hlir::framework::pir::Group* group;

  bool operator==(const DynamicTensor& other) const {
    return this->node_data == other.node_data;
  }

  std::size_t GetRank() const {
    return cinn::hlir::framework::pir::CompatibleInfo::ValueShape(node_data)
        .size();
  }

  std::vector<DimExpr> GetShape() const {
    std::vector<DimExpr> ret{};
    for (const auto& dim_expr : group->shape_analysis->GetShapeOrDataForValue(node_data).shape()) {
      ret.emplace_back(ConvertDimExpr(dim_expr));
    }
    return ret;
  }

  DimExpr ConvertDimExpr(const ::symbol::DimExpr& dim_expr) const {
    LOG(FATAL) << "TODO";
  }

};

inline std::size_t GetHashValueImpl(const DynamicTensor& tensor) {
  return std::hash<::pir::Value>()(tensor.node_data);
}

}  // namespace cinn::adt::adapter
