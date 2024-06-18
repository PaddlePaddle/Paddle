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
#include "paddle/cinn/adt/dim_expr.h"
#include "paddle/cinn/adt/symbolic_dim.h"
#include "paddle/cinn/hlir/framework/pir/op_lowering_group.h"

namespace cinn::adt::adapter {

struct DynamicTensor final {
  ::pir::Value node_data;
  const hlir::framework::pir::OpLoweringGroup* group;

  bool operator==(const DynamicTensor& other) const {
    return this->node_data == other.node_data;
  }

  std::size_t GetRank() const {
    return cinn::hlir::framework::pir::CompatibleInfo::ValueShape(node_data)
        .size();
  }

  const std::vector<DimExpr>& GetShape() const {
    return group->GetShapeOrDataExprs(node_data).shape();
  }
};

inline std::size_t GetHashValueImpl(const DynamicTensor& tensor) {
  return std::hash<::pir::Value>()(tensor.node_data);
}

}  // namespace cinn::adt::adapter
