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
using shape::SymbolicDimOp;

// Represents a product of symbolic and concrete factors.
// Used to prove product equalities symbolically.
struct SymbolicDimProduct {
  // List all symbolic factors that can not be aggregated.
  std::vector<SymbolicDimOp> symbols;

  // Product of all const factors.
  int64_t factor = 1;
  bool empty() { return factor == 1 && symbols.empty(); }
};

// Returns true if two SymbolicDimProduct are equal
inline bool operator==(const SymbolicDimProduct& lhs,
                       const SymbolicDimProduct& rhs) {
  return lhs.factor == rhs.factor && lhs.symbols == rhs.symbols;
}

// Returns true if two SymbolicDimProduct are not equal
inline bool operator!=(const SymbolicDimProduct& lhs,
                       const SymbolicDimProduct& rhs) {
  return !(lhs == rhs);
}

struct SymDimHasher {
  size_t operator()(const SymbolicDimOp& symbol) const noexcept {
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

// A class to manage shape-constraint related IR
class SymbolicDimMgr {
 public:
  explicit SymbolicDimMgr(ModuleOp m);

  // Loads pre-defined SymbolicDimOp ops from the module this mgr runs on.
  bool Load();

  // Create a new symbolicDim instance owned by this mgr.
  SymbolicDimOp NewSymbolicDim(const std::string& name = {});

  // Create a symbolicDim with static dim size == `val`.
  SymbolicDimOp NewConstantSymbolicDim(int64_t val);

  // Create a symbolicDim with given value.
  std::vector<SymbolicDimOp> CreateSymbolicDimsForRankedValue(Value value);

  // All symbolic-equal dims form a group.
  // Returns the root SymbolicDimOp of the symbolic-equal symbolic dim group
  // which this SymbolicDimOp belongs to.
  SymbolicDimOp GetRootSymbolicDim(SymbolicDimOp symbol);

  // Returns true if lhs and rhs are known to be equal.
  bool IsSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs);

  // Marks lhs and rhs have same size and try to merge lhs & rhs static known
  // info. Returns false if failed to merge lhs & rhs.
  bool MapSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs);

  // Returns the simplified version of SymbolicDimProduct.
  // This will try to fold some symbolicDim ops with const values.
  SymbolicDimProduct SimplifySymbolicDimProduct(const SymbolicDimProduct& x);

  // Returns the simplified version of SymbolicDimProductPair.
  // This will try to reduce some common symbolic ops if they are known nonzero.
  std::pair<SymbolicDimProduct, SymbolicDimProduct>
  SimplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                 const SymbolicDimProduct& y);

  // Returns null if x is not divided exactly by y, otherwise the result of x /
  // y Suppose that all symbols are nonzero, thus common symbolic dim factors
  // can be elimiated safely. For example:
  //    x = 6 * symbol_0 * symbol_1 * symbol_2
  //    y = 3 * symbol_0 * symbol_1
  //    x / y == 2 * symbol_2 (all symbols are nonzero)
  SymbolicDimProduct* SymbolicDimProductDivide(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y);

  // Mark group [a0, b0, ...] and [a1, b1, ...] are multiplication equal :
  //    `a0 * b0 * ... = a1 * b1 * c1 * ...`
  bool IsSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                 const SymbolicDimProduct& rhs);

  // Mark `product([a0, b0, ...]) == product([a1, b1, c1, ...])`
  bool MapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                  const SymbolicDimProduct& rhs);

  // Saves the updated shape constraint IR
  bool Save();

  // retuns the SymbolTable.
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

  std::unordered_map<SymbolicDimOp, SymbolicDimOp, SymDimHasher>
      symbol_dim_union_set_;

  std::unordered_map<int64_t, SymbolicDimOp> constant_symbolic_dim_map_;

  // product_equality_map_[A][B] == true : Product[A] == Product[B]
  using SymbolicDimProductMap = std::unordered_map<
      SymbolicDimProduct,
      std::unordered_map<SymbolicDimProduct, bool, SymProductHasher>,
      SymProductHasher>;
  SymbolicDimProductMap product_equality_map_;
  bool product_equality_map_updated_ = true;
};

}  // namespace pir
