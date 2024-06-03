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
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace cinn {
namespace dialect {
struct SymbolBindingBase {
  std::string symbol_name;
  int64_t input_tensor_idx;
  int64_t input_tensor_dim_idx;
  bool operator==(const SymbolBindingBase& other) const {
    return symbol_name == other.symbol_name &&
           input_tensor_idx == other.input_tensor_idx &&
           input_tensor_dim_idx == other.input_tensor_dim_idx;
  }
};

constexpr char* kDataSymbolBinding = "DataSymbolBinding";
constexpr char* kShapeSymbolBinding = "ShapeSymbolBinding";

struct DataSymbolBinding : public SymbolBindingBase {
  const char* binding_type() const { return kDataSymbolBinding; }
};
struct ShapeSymbolBinding : public SymbolBindingBase {
  const char* binding_type() const { return kShapeSymbolBinding; }
};

using SymbolBinding = std::variant<DataSymbolBinding, ShapeSymbolBinding>;

using SymbolBindings = std::vector<SymbolBinding>;

inline std::ostream& operator<<(std::ostream& os,
                                const SymbolBinding& symbol_binding) {
  std::visit(
      [&](auto&& binding) {
        os << binding.binding_type() << "[" << binding.symbol_name << ","
           << binding.input_tensor_idx << "," << binding.input_tensor_dim_idx
           << "]";
      },
      symbol_binding);
}
}  // namespace dialect
}  // namespace cinn
