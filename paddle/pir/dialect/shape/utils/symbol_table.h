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

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/builtin_type_interfaces.h"
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace pir {

using shape::SymbolicDimOp;
class SymbolTable {
 public:
  explicit SymbolTable(Operation* symbol_table_op)
      : symbol_table_op_(symbol_table_op) {}
  SymbolTable() = default;
  template <typename T>
  typename std::enable_if<std::is_same<T, SymbolicDimOp>::value,
                          SymbolicDimOp>::type
  Lookup(const std::string& name) const {
    auto it = symbol_table_map_.find(name);
    return it != symbol_table_map_.end() ? it->second->dyn_cast<SymbolicDimOp>()
                                         : SymbolicDimOp(nullptr);
  }
  template <typename T>
  typename std::enable_if<!std::is_same<T, SymbolicDimOp>::value,
                          std::vector<T>>::type
  Lookup(const std::string& name) const {
    std::vector<T> res;
    auto it = symbol_func_map_.find(name);
    if (it != symbol_func_map_.end()) {
      for (auto& p : it->second) {
        res.push_back(p->dyn_cast<T>());
      }
    }
    return res;
  }

  const std::string insert(Operation* symbol);
  Operation* getOp() const { return symbol_table_op_; }

 private:
  Operation* symbol_table_op_;
  std::unordered_map<std::string, Operation*> symbol_table_map_;
  std::unordered_map<std::string, std::vector<Operation*>> symbol_func_map_;
};

}  // namespace pir
