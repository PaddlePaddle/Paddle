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

// Helper class to query and manipulate shape constraint IR on buffer level.
class ShapeAnalysis {
 public:
  virtual ~ShapeAnalysis() = default;

  // Returns true if the two value have the same symbolic shape.
  virtual bool IsShapeEqual(Value lhs, Value rhs) = 0;

  // Suppose:
  //    lhs_dim_idxs = {ld0, ld1, ...}
  //    rhs_dim_idxs = {rd0, rd1, ...}
  // Returns true if:
  //    lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  //    rhs.shape[rd0] * rhs.shape[rd1] * ...
  virtual bool IsProductEqual(Value lhs,
                              std::vector<int> lhs_dim_idxs,
                              Value rhs,
                              std::vector<int> rhs_dim_idxs) = 0;

  // Returns true if:
  //    lhs.shape[lhs_from] * ... lhs.shape[lhs_to-1] ==
  //    rhs.shape[rhs_from] * ... rhs.shape[rhs_to-1]
  virtual bool IsProductEqual(
      Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to);

  // Returns true if the two value have the same number elements.
  virtual bool IsSameNumElements(Value lhs, Value rhs);
};

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

class SymbolTable {
 public:
  explicit SymbolTable(Operation* symbolTableOp)
      : symbolTableOp_(symbolTableOp) {}
  SymbolTable() = default;
  template <typename T>
  typename std::enable_if<std::is_same<T, SymbolicDim>::value,
                          SymbolicDim>::type
  Lookup(const std::string& name) const {
    auto it = symbolTableMap_.find(name);
    return it != symbolTableMap_.end() ? it->second->dyn_cast<SymbolicDim>()
                                       : SymbolicDim(nullptr);
  }
  template <typename T>
  typename std::enable_if<!std::is_same<T, SymbolicDim>::value,
                          std::vector<T>>::type
  Lookup(const std::string& name) const {
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
  Operation* getOp() const { return symbolTableOp_; }

 private:
  Operation* symbolTableOp_;
  std::unordered_map<std::string, Operation*> symbolTableMap_;
  std::unordered_map<std::string, std::vector<Operation*>> symbolFuncMap_;
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
  SymbolTable& symbolTable() { return symbolTable_; }
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

 private:
  const std::string GetNextName();
  bool UpdateProductEqualityMap();
  bool IsMultipleOfKnownSymbolicDimProductEqualPair(
      const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs);
  bool SaveShapeConstraintGraph();
  bool LoadShapeConstraintGraph();

 private:
  ModuleOp m_;

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

// A subclass to impement `ShapeAnalysis` on buffer level.
// The implementation is based on shape constraint ir.
class ShapeConstraintIRAnalysis : public ShapeAnalysis {
 public:
  // Build shape related analysis on the provided `op`.
  // This generally can be divided into two steps:
  // 1, load exsiting shape constraint ir (e.g. symbolic dim ops)
  // 2, build mapping between memref values and symbolic dim ops.
  explicit ShapeConstraintIRAnalysis(ModuleOp m);

  // auto-save updated shape constriant ir when destroying.
  ~ShapeConstraintIRAnalysis();

  // Returns the `SymbolicDimMgr` this object holds.
  SymbolicDimMgr& symbolicDimMgr() { return mgr_; }
  const SymbolicDimMgr& symbolicDimMgr() const { return mgr_; }

  // Returns true if the two value have the same symbolic shape.
  bool IsShapeEqual(Value lhs, Value rhs) override;

  // Suppose:
  //    lhs_dim_idxs = {ld0, ld1, ...}
  //    rhs_dim_idxs = {rd0, rd1, ...}
  // Returns true if:
  //    lhs.shape[ld0] * lhs.shape[ld1] * ... ==
  //    rhs.shape[rd0] * rhs.shape[rd1] * ...
  bool IsProductEqual(Value lhs,
                      std::vector<int> lhs_dim_idxs,
                      Value rhs,
                      std::vector<int> rhs_dim_idxs) override;

 private:
  // The operation this analysis runs on.
  ModuleOp m_;
  // The `SymbolicDimMgr` this analysis holds.
  SymbolicDimMgr mgr_;
  // Map a ranked memref value to an array of symbolicDims, each represents one
  // dimension size of the memref value.
  std::unordered_map<Value, std::vector<SymbolicDim>> value_to_sym_dims_;
};

class ShapeComputationIRAnalysis {
 public:
  using func = std::function<bool(Operation* op)>;
  explicit ShapeComputationIRAnalysis(ModuleOp m,
                                      SymbolicDimMgr& mgr);  // NOLINT
  bool Run();

 private:
  bool RunOnRegion(Region* region, func fn);
  bool RunOnBlock(Block* block, func fn);
  bool RunOnOperation(Operation* op, func fn);

  bool BuildShapeOnOperation(Operation* op);
  bool BuildShapeOnValue(Value value);

  bool ApplyOpConstraint(Operation* op);
  bool ApplyIndexOpConstraint(Operation* op);
  bool ApplyTieShapeOpConstraint(Operation* op);

  bool initialized_ = false;
  ModuleOp m_;
  SymbolicDimMgr& mgr_;

  std::unordered_map<Value, SymbolicDim> value2SymDim_;

  // shape tensor is the 1D ranked tensor with int/index dtype.
  std::unordered_map<Value, std::vector<SymbolicDim>> shapeTensor2SymDims_;

  std::unordered_map<Value, std::vector<SymbolicDim>> rankedTensor2SymDims_;
};

bool IsIntOrIndex(Type type);
bool IsCandidateShapeTensorType(Type ty);
}  // namespace pir
