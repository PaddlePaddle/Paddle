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

#include "paddle/ir/dialect/shape/utils/shape_utils.h"
#include <string>
#include "paddle/fluid/ir/dialect/paddle_dialect/ir/pd_type.h"
namespace ir {

bool compareSymbolicDimNames(const std::string& lhs, const std::string& rhs) {
  if (lhs.size() < 1 || (lhs[0] != 'S' && lhs[0] != 'C')) return lhs < rhs;
  if (rhs.size() < 1 || (rhs[0] != 'S' && rhs[0] != 'C')) return lhs < rhs;
  int64_t lhsIdx = 0, rhsIdx = 0;
  try {
    lhsIdx = stol(lhs.substr(1));
    rhsIdx = stol(rhs.substr(1));
  } catch (const std::exception& e) {
    IR_THROW("Invalid symbolic name");
  }
  return (lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhsIdx < rhsIdx);
}

const std::string SymbolTable::insert(ir::Operation* symbol) {
  std::string name;
  if (symbol->name() == "shape.SymbolicDim") {
    name = symbol->dyn_cast<SymbolicDim>().getSymName();
    symbolTableMap_.insert({name, symbol});
  }

  // TODO(liujinnan): add more constraint_func name branch.
  if (symbol->name() == "shape.tie_product_equal") {
    name = "tie_product_equal";
    symbolFuncMap_[name].emplace_back(symbol);
  }

  return name;
}

bool SymbolicDimMgr::load() {
  for (auto op_it = m_.block()->begin(); op_it != m_.block()->end(); op_it++) {
    symbolTable_.insert(*op_it);
    SymbolicDim op = (*op_it)->dyn_cast<SymbolicDim>();
    if (!op) continue;
    symbolDimUnionSet_[op] = op;
    symbolNameSet_.insert(op.getSymName());
  }
  return loadShapeConstraintGraph();
}

bool SymbolicDimMgr::loadShapeConstraintGraph() {
  // TODO(liujinnan): add more constraint function. currently, only support
  // tie_product_equal.
  auto constraint_vec =
      symbolTable_.lookup<ir::dialect::TieProductEqualOp>("tie_product_equal");

  if (!constraint_vec.size()) return true;

  auto build_sym_product = [&](std::vector<ir::Value> range,
                               SymbolicDimProduct& product) {
    for (Value v : range) {
      auto definingOp = v.GetDefiningOp();
      if (auto constOp = definingOp->dyn_cast<ir::ConstantOp>()) {
        product.factor *= constOp.value().dyn_cast<ir::Int32Attribute>().data();
        continue;
      } else if (auto dimOp = definingOp->dyn_cast<ir::dialect::DimOp>()) {
        auto sym = symbolTable_.lookup<SymbolicDim>(dimOp.getName());
        if (!sym) return false;
        product.symbols.push_back(sym);
        continue;
      }
      return false;
    }
    return true;
  };
  for (auto op : constraint_vec) {
    SymbolicDimProduct lhs, rhs;
    if (!build_sym_product(op.getLhs(), lhs) ||
        !build_sym_product(op.getRhs(), rhs) ||
        !mapSymbolicDimProductEqual(lhs, rhs))
      return false;
  }
  return true;
}

int64_t gcd(int64_t m, int64_t n) {
  if (!m) return n;
  if (!n) return m;
  return (m < n) ? gcd(m, n % m) : gcd(m % n, n);
}

bool SymbolicDimMgr::mapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                                const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (newLhs == newRhs) return true;

  if (newLhs.factor == newRhs.factor && newLhs.symbols.size() == 1 &&
      newRhs.symbols.size() == 1) {
    return mapSymbolicDimEqual(newLhs.symbols[0], newRhs.symbols[0]);
  } else if (newLhs.symbols.size() == 0 && newRhs.symbols.size() == 1 &&
             newRhs.factor == 1) {
    return mapSymbolicDimEqual(newConstantSymbolicDim(newLhs.factor),
                               newRhs.symbols[0]);
  } else if (newRhs.symbols.size() == 0 && newLhs.symbols.size() == 1 &&
             newLhs.factor == 1) {
    return mapSymbolicDimEqual(newConstantSymbolicDim(newRhs.factor),
                               newLhs.symbols[0]);
  }

  productEqualityMap_[newLhs][newRhs] = productEqualityMap_[newRhs][newLhs] =
      true;

  productEqualityMapUpdated_ = false;
  return true;
}

std::pair<SymbolicDimProduct, SymbolicDimProduct>
SymbolicDimMgr::simplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y) {
  auto lhs = simplifySymbolicDimProduct(x);
  auto rhs = simplifySymbolicDimProduct(y);

  SymbolicDimProduct newLhs, newRhs;
  int64_t gcdFactor = gcd(std::abs(lhs.factor), std::abs(rhs.factor));
  if (!gcdFactor) return std::make_pair(std::move(newLhs), std::move(newRhs));
  if (std::abs(lhs.factor) < std::abs(rhs.factor)) {
    if (lhs.factor < 0) gcdFactor = -gcdFactor;
  } else {
    if (rhs.factor < 0) gcdFactor = -gcdFactor;
  }

  newLhs.factor = lhs.factor / gcdFactor;
  newRhs.factor = rhs.factor / gcdFactor;

  std::unordered_map<SymbolicDim, int, SymDimHasher> lhsSymbolMap;
  std::unordered_map<SymbolicDim, int, SymDimHasher> rhsSymbolMap;
  for (SymbolicDim op : lhs.symbols) ++lhsSymbolMap[op];
  for (SymbolicDim op : rhs.symbols) ++rhsSymbolMap[op];

  for (SymbolicDim op : lhs.symbols) {
    auto it = rhsSymbolMap.find(op);
    if (it != rhsSymbolMap.end() && op.getKnownNonSizeZero()) {
      if (--it->second == 0) rhsSymbolMap.erase(it);
      continue;
    }
    newLhs.symbols.push_back(op);
  }

  for (SymbolicDim op : rhs.symbols) {
    auto it = lhsSymbolMap.find(op);
    if (it != lhsSymbolMap.end() && op.getKnownNonSizeZero()) {
      if (--it->second == 0) lhsSymbolMap.erase(it);
      continue;
    }
    newRhs.symbols.push_back(op);
  }

  if (!newLhs.factor) newLhs.symbols.clear();
  if (!newRhs.factor) newRhs.symbols.clear();

  return std::make_pair(std::move(newLhs), std::move(newRhs));
}

SymbolicDimProduct SymbolicDimMgr::simplifySymbolicDimProduct(
    const SymbolicDimProduct& x) {
  std::vector<SymbolicDim> copied;
  copied.reserve(x.symbols.size());
  for (SymbolicDim op : x.symbols) copied.push_back(getRootSymbolicDim(op));

  sort(copied.begin(), copied.end(), [&](SymbolicDim lhs, SymbolicDim rhs) {
    return compareSymbolicDimNames(lhs.getSymName(), rhs.getSymName());
  });
  SymbolicDimProduct newX;
  newX.factor = x.factor;
  for (SymbolicDim op : copied) {
    if (!op.isDynamic()) {
      newX.factor *= op.getValue();
    } else {
      newX.symbols.push_back(op);
    }
  }
  return newX;
}

const std::string SymbolicDimMgr::getNextName() {
  std::string name;
  do {
    name = "S" + std::to_string(nextSymbolicIdx_++);
  } while (!symbolNameSet_.insert(name).second);
  return name;
}

SymbolicDimMgr::SymbolicDimMgr(ir::ModuleOp m) : m_(m), symbolTable_(m_) {}

SymbolicDim SymbolicDimMgr::newSymbolicDim(const std::string& name) {
  ::ir::Builder builder = ::ir::Builder(m_.ir_context(), m_.block());
  ir::dialect::SymbolicDim symbol = builder.Build<ir::dialect::SymbolicDim>(
      name.empty() ? getNextName() : name);
  symbolDimUnionSet_[symbol] = symbol;
  symbolTable_.insert(symbol);
  return symbol;
}

SymbolicDim SymbolicDimMgr::newConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    auto name = "C" + std::to_string(val);
    it = constantSymbolicDimMap_
             .insert(std::make_pair(val, newSymbolicDim(name)))
             .first;
    it->second.updateValue(val);
  }
  return getRootSymbolicDim(it->second);
}

std::vector<SymbolicDim> SymbolicDimMgr::createSymbolicDimsForRankedValue(
    ir::Value value) {
  std::vector<SymbolicDim> symbols;
  auto dims = value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
  for (int idx = 0; idx < dims.size(); ++idx) {
    symbols.push_back(
        dims[idx] == -100000  // TODO(zhangbo): value = ShapedType::kDynamic
            ? newSymbolicDim()
            : newConstantSymbolicDim(dims[idx]));
  }
  return symbols;
}

SymbolicDim SymbolicDimMgr::getRootSymbolicDim(SymbolicDim symbol) {
  SymbolicDim current = symbol;
  std::vector<SymbolicDim> path;
  while (symbolDimUnionSet_[current] != current) {
    path.push_back(current);
    current = symbolDimUnionSet_[current];
  }
  for (SymbolicDim sym : path) symbolDimUnionSet_[sym] = current;
  return current;
}

bool SymbolicDimMgr::isSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs) {
  SymbolicDim lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDim rhsRoot = getRootSymbolicDim(rhs);
  return lhsRoot == rhsRoot;
}

bool SymbolicDimMgr::mapSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs) {
  SymbolicDim lhsRoot = getRootSymbolicDim(lhs);
  SymbolicDim rhsRoot = getRootSymbolicDim(rhs);

  if (lhsRoot != rhsRoot) {
    if (compareSymbolicDimNames(lhsRoot.getSymName(), rhsRoot.getSymName())) {
      if (!lhsRoot.merge(rhsRoot)) return false;
      symbolDimUnionSet_[rhsRoot] = lhsRoot;
    } else {
      if (!rhsRoot.merge(lhsRoot)) return false;
      symbolDimUnionSet_[lhsRoot] = rhsRoot;
    }
  }
  return true;
}

SymbolicDimProduct* SymbolicDimMgr::symbolicDimProductDivide(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);

  if (newLhs.factor == 0 || newRhs.factor == 0) return nullptr;
  if (newLhs.factor % newRhs.factor != 0) return nullptr;
  if (newLhs.symbols.size() < newRhs.symbols.size()) return nullptr;

  SymbolicDimProduct* result = new SymbolicDimProduct();
  result->factor = newLhs.factor / newRhs.factor;

  std::unordered_map<SymbolicDim, int, SymDimHasher> symProcMap;
  for (SymbolicDim sym : newRhs.symbols) ++symProcMap[sym];

  for (SymbolicDim sym : newLhs.symbols) {
    auto it = symProcMap.find(sym);
    if (it == symProcMap.end()) {
      result->symbols.push_back(sym);
      continue;
    }
    if (--it->second == 0) {
      symProcMap.erase(it);
      continue;
    }
  }

  if (!symProcMap.empty()) return nullptr;
  return result;
}

bool SymbolicDimMgr::isMultipleOfKnownSymbolicDimProductEqualPair(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  for (auto& pairOutter : productEqualityMap_) {
    const SymbolicDimProduct& x = pairOutter.first;
    auto factorX = symbolicDimProductDivide(lhs, x);
    if (!factorX) continue;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      const SymbolicDimProduct& y = pairInner.first;
      auto factorY = symbolicDimProductDivide(rhs, y);
      if (!factorY || (*factorX) != (*factorY)) continue;
      return true;
    }
  }

  return false;
}

bool SymbolicDimMgr::updateProductEqualityMap() {
  // early return if nothing is updated.
  if (productEqualityMapUpdated_) return true;

  SymbolicDimProductMap newMap;
  std::unordered_set<SymbolicDimProduct, SymProductHasher> productSet;
  for (auto& pairOutter : productEqualityMap_) {
    const SymbolicDimProduct& x = pairOutter.first;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      const SymbolicDimProduct& y = pairInner.first;
      SymbolicDimProduct newX, newY;
      std::tie(newX, newY) = simplifySymbolicDimProductPair(x, y);
      if (newX == newY) continue;
      newMap[newX][newY] = newMap[newY][newX] = true;
      productSet.insert(newX);
      productSet.insert(newY);
    }
  }
  // hash function of SymbolicDimProduct is expensive, thus we map it to integer
  // domain first.
  std::unordered_map<const SymbolicDimProduct*, size_t> symProd2Idx;
  std::vector<const SymbolicDimProduct*> idx2SymProd(productSet.size());
  std::vector<size_t> idx2root(productSet.size());
  for (auto& x : productSet) {
    size_t idx = symProd2Idx.size();
    symProd2Idx[&x] = idx;
    idx2SymProd[idx] = &x;
    idx2root[idx] = idx;
  }

  auto getRootIdx = [&](size_t root) {
    std::vector<size_t> path;
    while (idx2root[root] != root) {
      path.push_back(root);
      root = idx2root[root];
    }
    for (size_t idx : path) idx2root[idx] = root;
    return root;
  };

  for (size_t x = 0; x < symProd2Idx.size(); ++x) {
    auto& xProd = *idx2SymProd[x];
    auto& rowMap = newMap[xProd];
    size_t xRoot = getRootIdx(x);
    for (size_t y = x; y < symProd2Idx.size(); ++y) {
      auto& yProd = *idx2SymProd[y];
      if (!rowMap[yProd]) continue;
      idx2root[getRootIdx(y)] = xRoot;
    }
  }

  for (size_t x = 0; x < symProd2Idx.size(); ++x)
    for (size_t y = x; y < symProd2Idx.size(); ++y) {
      if (getRootIdx(x) != getRootIdx(y)) continue;
      auto& xSymProd = *idx2SymProd[x];
      auto& ySymProd = *idx2SymProd[y];

      newMap[xSymProd][ySymProd] = newMap[ySymProd][xSymProd] = true;
    }

  productEqualityMap_ = std::move(newMap);

  for (auto& x : productSet)
    for (auto& y : productSet) {
      if (!productEqualityMap_[x][y]) continue;
      productEqualityMap_[x][y] = productEqualityMap_[y][x] = false;
      if (!isMultipleOfKnownSymbolicDimProductEqualPair(x, y)) {
        productEqualityMap_[x][y] = productEqualityMap_[y][x] = true;
      }
    }

  std::unordered_set<SymbolicDimProduct, SymProductHasher> toRemove;
  for (auto& x : productSet) {
    if (std::all_of(productSet.begin(),
                    productSet.end(),
                    [&](const SymbolicDimProduct& y) {
                      return !productEqualityMap_[x][y];
                    })) {
      toRemove.insert(x);
    }
  }

  for (auto& x : toRemove) {
    productEqualityMap_.erase(x);
  }

  productEqualityMapUpdated_ = true;
  return true;
}

bool SymbolicDimMgr::isSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                               const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (newLhs == newRhs) return true;
  IR_ENFORCE(updateProductEqualityMap(), "Update product equality map failed.");
  return isMultipleOfKnownSymbolicDimProductEqualPair(newLhs, newRhs);
}
}  // namespace ir
