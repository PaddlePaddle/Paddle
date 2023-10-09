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
#include <vector>
#include "paddle/pir/dialect/shape/utils/symbol_table.h"

namespace pir {
using dialect::SymbolicDim;

struct SymbolicDimProduct {
  std::vector<SymbolicDim> symbols;
  int64_t factor = 1;
  bool empty() { return factor == 1 && symbols.empty(); }
  friend inline bool operator==(const SymbolicDimProduct& lhs,
                                const SymbolicDimProduct& rhs) {
    return lhs.factor == rhs.factor && lhs.symbols == rhs.symbols;
  }

  friend inline bool operator!=(const SymbolicDimProduct& lhs,
                                const SymbolicDimProduct& rhs) {
    return !(lhs == rhs);
  }
};

struct SymDimHasher {
  size_t operator()(const dialect::SymbolicDim& symbol) const noexcept {
    return std::hash<Operation*>{}(symbol.operation());
  }
};

struct SymProductHasher {
  size_t operator()(const SymbolicDimProduct& symProd) const noexcept {
    size_t hash = std::hash<size_t>{}(symProd.symbols.size());
    for (auto& symbol : symProd.symbols) {
      hash = hash_combine(hash, SymDimHasher{}(symbol));  // NOLINT
    }
    hash = hash_combine(hash, std::hash<int64_t>{}(symProd.factor));
    return hash;
  }
};

class SymbolicDimMgr {
 public:
  explicit SymbolicDimMgr(ModuleOp m);
  bool Load();
  SymbolicDim NewSymbolicDim(const std::string& name = {});
  SymbolicDim NewConstantSymbolicDim(int64_t val);
  std::vector<SymbolicDim> CreateSymbolicDimsForRankedValue(Value value);
  SymbolicDim GetRootSymbolicDim(SymbolicDim symbol);
  bool IsSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs);
  bool MapSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs);
  SymbolicDimProduct SimplifySymbolicDimProduct(const SymbolicDimProduct& x);
  std::pair<SymbolicDimProduct, SymbolicDimProduct>
  SimplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                 const SymbolicDimProduct& y);
  SymbolicDimProduct* SymbolicDimProductDivide(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y);
  bool Save();
  bool IsSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                 const SymbolicDimProduct& rhs);

  bool MapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                  const SymbolicDimProduct& rhs);
  SymbolTable& symbolTable() { return symbol_table_; }

 private:
  const std::string GetNextName();
  bool SaveShapeConstraintGraph();
  bool LoadShapeConstraintGraph();
  bool UpdateProductEqualityMap();
  bool IsMultipleOfKnownSymbolicDimProductEqualPair(
      const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs);

 private:
  ModuleOp m_;

  SymbolTable symbol_table_;

  int64_t next_symbolic_idx_ = 0;

  std::unordered_set<std::string> symbol_name_set_;

  std::unordered_map<SymbolicDim, SymbolicDim, SymDimHasher>
      symbol_dim_union_set_;

  std::unordered_map<int64_t, SymbolicDim> constant_symbolic_dim_map_;

  // product_equality_map_[A][B] == true : Product[A] == Product[B]
  using SymbolicDimProductMap = std::unordered_map<
      SymbolicDimProduct,
      std::unordered_map<SymbolicDimProduct, bool, SymProductHasher>,
      SymProductHasher>;
  SymbolicDimProductMap product_equality_map_;
  bool product_equality_map_updated_ = true;
};

}  // namespace pir
