// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#include <absl/container/flat_hash_map.h>
#include <absl/types/variant.h>
#include <string>
#include <vector>
#include "paddle/cinn/hlir/dialect/operator/ir/symbol_bindings.h"
#include "paddle/pir/include/dialect/shape/utils/dim_expr.h"

namespace cinn {
namespace utils {

// attribute type defs
using Attribute = absl::variant<bool,
                                float,
                                int,
                                std::string,
                                std::vector<bool>,
                                std::vector<int>,
                                std::vector<float>,
                                std::vector<std::string>,
                                int64_t,
                                double,
                                std::vector<int64_t>,
                                std::vector<double>,
                                // the followings are only for generate shape op
                                std::vector<symbol::DimExpr>,
                                cinn::dialect::SymbolBindings>;
using AttributeMap = absl::flat_hash_map<std::string, Attribute>;

// shape type defs
using ShapeType = std::vector<int32_t>;
using DimType = ShapeType::value_type;

}  // namespace utils
}  // namespace cinn
