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
#include "paddle/pir/core/utils.h"
#include "paddle/pir/dialect/shape/ir/shape_op.h"

namespace pir {

using pir::dialect::SymbolicDim;

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

class SymbolTable {
 public:
  explicit SymbolTable(pir::Operation* symbolTableOp)
      : symbolTableOp_(symbolTableOp) {}
  SymbolTable() = default;
  template <typename T>
  typename std::enable_if<std::is_same<T, SymbolicDim>::value,
                          SymbolicDim>::type
  lookup(const std::string& name) const {
    auto it = symbolTableMap_.find(name);
    return it != symbolTableMap_.end() ? it->second->dyn_cast<SymbolicDim>()
                                       : SymbolicDim(nullptr);
  }
  template <typename T>
  typename std::enable_if<!std::is_same<T, SymbolicDim>::value,
                          std::vector<T>>::type
  lookup(const std::string& name) const {
    std::vector<T> res;
    auto it = symbolFuncMap_.find(name);
    if (it != symbolFuncMap_.end()) {
      for (auto& p : it->second) {
        res.push_back(p->dyn_cast<T>());
      }
    }
    return res;
  }

  const std::string insert(Operation* symbol);
  pir::Operation* getOp() const { return symbolTableOp_; }

 private:
  pir::Operation* symbolTableOp_;
  std::unordered_map<std::string, pir::Operation*> symbolTableMap_;
  std::unordered_map<std::string, std::vector<pir::Operation*>> symbolFuncMap_;
};

struct SymDimHasher {
  size_t operator()(const pir::dialect::SymbolicDim& symbol) const noexcept {
    return std::hash<pir::Operation*>{}(symbol.operation());
  }
};

struct SymProductHasher {
  size_t operator()(const pir::SymbolicDimProduct& symProd) const noexcept {
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
  explicit SymbolicDimMgr(pir::ModuleOp m);
  bool load();
  SymbolicDim newSymbolicDim(const std::string& name = {});
  SymbolicDim newConstantSymbolicDim(int64_t val);
  std::vector<SymbolicDim> createSymbolicDimsForRankedValue(Value value);
  SymbolicDim getRootSymbolicDim(SymbolicDim symbol);
  bool isSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs);
  SymbolTable& symbolTable() { return symbolTable_; }
  bool mapSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs);
  SymbolicDimProduct simplifySymbolicDimProduct(const SymbolicDimProduct& x);
  std::pair<SymbolicDimProduct, SymbolicDimProduct>
  simplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                 const SymbolicDimProduct& y);
  SymbolicDimProduct* symbolicDimProductDivide(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y);

  bool save();

  bool isSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                 const SymbolicDimProduct& rhs);
  bool mapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                  const SymbolicDimProduct& rhs);

 private:
  const std::string getNextName();
  bool updateProductEqualityMap();
  bool isMultipleOfKnownSymbolicDimProductEqualPair(
      const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs);
  bool saveShapeConstraintGraph();
  bool loadShapeConstraintGraph();

 private:
  pir::ModuleOp m_;

  SymbolTable symbolTable_;

  int64_t nextSymbolicIdx_ = 0;

  std::unordered_set<std::string> symbolNameSet_;

  std::unordered_map<SymbolicDim, SymbolicDim, SymDimHasher> symbolDimUnionSet_;

  std::unordered_map<int64_t, SymbolicDim> constantSymbolicDimMap_;

  // productEqualityMap_[A][B] == true : Product[A] == Product[B]
  using SymbolicDimProductMap = std::unordered_map<
      SymbolicDimProduct,
      std::unordered_map<SymbolicDimProduct, bool, SymProductHasher>,
      SymProductHasher>;
  SymbolicDimProductMap productEqualityMap_;
  bool productEqualityMapUpdated_ = true;
};

class ShapeAnalysis {
 public:
  virtual ~ShapeAnalysis() = default;

  virtual bool isShapeEqual(pir::Value lhs, pir::Value rhs) = 0;

  virtual bool isProductEqual(pir::Value lhs,
                              std::vector<int> lhsDimIdxs,
                              pir::Value rhs,
                              std::vector<int> rhsDimIdxs) = 0;
  virtual bool isProductEqual(pir::Value lhs,
                              int lhsFrom,
                              int lhsTo,
                              pir::Value rhs,
                              int rhsFrom,
                              int rhsTo);
  virtual bool isSameNumElements(pir::Value lhs, pir::Value rhs);
};

class SymbolicDimShapeAnalysis : public ShapeAnalysis {
 public:
  explicit SymbolicDimShapeAnalysis(pir::Operation* op);
  ~SymbolicDimShapeAnalysis();

  SymbolicDimMgr& symbolicDimMgr() { return mgr_; }
  const SymbolicDimMgr& symbolicDimMgr() const { return mgr_; }
  bool isShapeEqual(pir::Value lhs, pir::Value rhs) override;

  bool isProductEqual(pir::Value lhs,
                      std::vector<int> lhsDimIdxs,
                      pir::Value rhs,
                      std::vector<int> rhsDimIdxs) override;

 private:
  pir::Operation* op_;
  SymbolicDimMgr mgr_;
  std::unordered_map<pir::Value, std::vector<SymbolicDim>> value2SymDims_;
};
}  // namespace pir
