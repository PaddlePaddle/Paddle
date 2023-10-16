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

#include "paddle/pir/dialect/shape/utils/symbol_table.h"

namespace pir {

const std::string SymbolTable::insert(Operation* symbol) {
  std::string name;
  if (symbol->isa<shape::SymbolicDimOp>()) {
    name = symbol->dyn_cast<SymbolicDimOp>().GetSymName();
    symbol_table_map_.insert({name, symbol});
  }

  // TODO(zhangbopd): add more constraint_func name branch.
  if (symbol->isa<shape::TieProductEqualOp>()) {
    name = "tie_product_equal";
    symbol_func_map_[name].emplace_back(symbol);
  }

  return name;
}
}  // namespace pir
