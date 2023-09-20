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

#include "paddle/pir/dialect/shape/utils/shape_utils.h"
#include <string>
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
namespace pir {

bool CompareSymbolicDimNames(const std::string& lhs, const std::string& rhs) {
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

bool CompareSymbolicDimProduct(SymbolicDimProduct& lhs,    // NOLINT
                               SymbolicDimProduct& rhs) {  // NOLINT
  if (lhs.symbols.size() < rhs.symbols.size()) return true;
  if (lhs.symbols.size() == rhs.symbols.size()) {
    for (size_t idx = 0; idx < lhs.symbols.size(); ++idx) {
      const std::string lhsName = lhs.symbols[idx].getSymName();
      const std::string rhsName = rhs.symbols[idx].getSymName();
      if (CompareSymbolicDimNames(lhsName, rhsName)) return true;
      if (lhsName != rhsName) return false;
    }
  }
  return false;
}

const std::string SymbolTable::insert(Operation* symbol) {
  std::string name;
  if (symbol->isa<dialect::SymbolicDim>()) {
    name = symbol->dyn_cast<SymbolicDim>().getSymName();
    symbolTableMap_.insert({name, symbol});
  }

  // TODO(liujinnan): add more constraint_func name branch.
  if (symbol->isa<dialect::TieProductEqualOp>()) {
    name = "tie_product_equal";
    symbolFuncMap_[name].emplace_back(symbol);
  }

  return name;
}

bool SymbolicDimMgr::Load() {
  auto funcOp = symbolTable_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(funcOp);
  for (auto op_ : *(funcOp.block())) {
    symbolTable_.insert(op_);
    if (SymbolicDim op = op_->dyn_cast<SymbolicDim>()) {
      symbolDimUnionSet_[op] = op;
      symbolNameSet_.insert(op.getSymName());
    }
  }
  return LoadShapeConstraintGraph();
}

bool SymbolicDimMgr::LoadShapeConstraintGraph() {
  // TODO(liujinnan): add more constraint function. currently, only support
  // tie_product_equal.
  auto constraint_vec =
      symbolTable_.Lookup<dialect::TieProductEqualOp>("tie_product_equal");

  if (!constraint_vec.size()) return true;

  auto build_sym_product = [&](std::vector<Value> range,
                               SymbolicDimProduct& product) {
    for (Value v : range) {
      auto definingOp = v.dyn_cast<OpResult>().owner();
      if (auto constOp = definingOp->dyn_cast<ConstantOp>()) {
        product.factor *= constOp.value().dyn_cast<Int32Attribute>().data();
        continue;
      } else if (auto dimOp = definingOp->dyn_cast<dialect::DimOp>()) {
        auto sym = symbolTable_.Lookup<SymbolicDim>(dimOp.getName());
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
    if (!build_sym_product(op.lhs(), lhs) ||
        !build_sym_product(op.rhs(), rhs) ||
        !MapSymbolicDimProductEqual(lhs, rhs))
      return false;
  }
  return true;
}

int64_t gcd(int64_t m, int64_t n) {
  if (!m) return n;
  if (!n) return m;
  return (m < n) ? gcd(m, n % m) : gcd(m % n, n);
}

bool SymbolicDimMgr::MapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                                const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (newLhs == newRhs) return true;

  if (newLhs.factor == newRhs.factor && newLhs.symbols.size() == 1 &&
      newRhs.symbols.size() == 1) {
    return MapSymbolicDimEqual(newLhs.symbols[0], newRhs.symbols[0]);
  } else if (newLhs.symbols.size() == 0 && newRhs.symbols.size() == 1 &&
             newRhs.factor == 1) {
    return MapSymbolicDimEqual(NewConstantSymbolicDim(newLhs.factor),
                               newRhs.symbols[0]);
  } else if (newRhs.symbols.size() == 0 && newLhs.symbols.size() == 1 &&
             newLhs.factor == 1) {
    return MapSymbolicDimEqual(NewConstantSymbolicDim(newRhs.factor),
                               newLhs.symbols[0]);
  }

  productEqualityMap_[newLhs][newRhs] = productEqualityMap_[newRhs][newLhs] =
      true;

  productEqualityMapUpdated_ = false;
  return true;
}

std::pair<SymbolicDimProduct, SymbolicDimProduct>
SymbolicDimMgr::SimplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y) {
  auto lhs = SimplifySymbolicDimProduct(x);
  auto rhs = SimplifySymbolicDimProduct(y);

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

SymbolicDimProduct SymbolicDimMgr::SimplifySymbolicDimProduct(
    const SymbolicDimProduct& x) {
  std::vector<SymbolicDim> copied;
  copied.reserve(x.symbols.size());
  for (SymbolicDim op : x.symbols) copied.push_back(GetRootSymbolicDim(op));

  sort(copied.begin(), copied.end(), [&](SymbolicDim lhs, SymbolicDim rhs) {
    return CompareSymbolicDimNames(lhs.getSymName(), rhs.getSymName());
  });
  SymbolicDimProduct newX;
  newX.factor = x.factor;
  for (SymbolicDim op : copied) {
    if (!op.IsDynamic()) {
      newX.factor *= op.getValue();
    } else {
      newX.symbols.push_back(op);
    }
  }
  return newX;
}

const std::string SymbolicDimMgr::GetNextName() {
  std::string name;
  do {
    name = "S" + std::to_string(nextSymbolicIdx_++);
  } while (!symbolNameSet_.insert(name).second);
  return name;
}

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) : m_(m) {
  for (auto op : *(m.block())) {
    if (op->isa<dialect::FuncOp>()) {
      symbolTable_ = SymbolTable(op);
      return;
    }
  }
  Builder builder = Builder(m_.ir_context(), m_.block(), m_.block()->begin());
  dialect::FuncOp func = builder.Build<dialect::FuncOp>();
  symbolTable_ = SymbolTable(func);
}

SymbolicDim SymbolicDimMgr::NewSymbolicDim(const std::string& name) {
  auto funcOp = symbolTable_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(funcOp);
  Builder builder = Builder(m_.ir_context(), funcOp.block());
  // default settting dim != 0
  dialect::SymbolicDim symbol =
      builder.Build<dialect::SymbolicDim>(name.empty() ? GetNextName() : name,
                                          ShapedTypeInterface::kDynamic,
                                          false,
                                          false,
                                          false,
                                          true);
  symbolDimUnionSet_[symbol] = symbol;
  symbolTable_.insert(symbol);
  return symbol;
}

SymbolicDim SymbolicDimMgr::NewConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    auto name = "C" + std::to_string(val);
    it = constantSymbolicDimMap_
             .insert(std::make_pair(val, NewSymbolicDim(name)))
             .first;
    it->second.updateValue(val);
    if (val == -1) it->second.updateKnownNegativeOne(true);
    if (val >= 0) it->second.updateKnownNonNegative(true);
    if (val != 1) it->second.updateKnownNonSizeOne(true);
    if (val != 0) it->second.updateKnownNonSizeZero(true);
  }
  return GetRootSymbolicDim(it->second);
}

std::vector<SymbolicDim> SymbolicDimMgr::CreateSymbolicDimsForRankedValue(
    Value value) {
  std::vector<SymbolicDim> symbols;
  auto dims = value.type().dyn_cast<paddle::dialect::DenseTensorType>().dims();
  for (int idx = 0; idx < dims.size(); ++idx) {
    symbols.push_back(dims[idx] == ShapedTypeInterface::kDynamic
                          ? NewSymbolicDim()
                          : NewConstantSymbolicDim(dims[idx]));
  }
  return symbols;
}

SymbolicDim SymbolicDimMgr::GetRootSymbolicDim(SymbolicDim symbol) {
  SymbolicDim current = symbol;
  std::vector<SymbolicDim> path;
  while (symbolDimUnionSet_[current] != current) {
    path.push_back(current);
    current = symbolDimUnionSet_[current];
  }
  for (SymbolicDim sym : path) symbolDimUnionSet_[sym] = current;
  return current;
}

bool SymbolicDimMgr::IsSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs) {
  SymbolicDim lhsRoot = GetRootSymbolicDim(lhs);
  SymbolicDim rhsRoot = GetRootSymbolicDim(rhs);
  return lhsRoot == rhsRoot;
}

bool SymbolicDimMgr::MapSymbolicDimEqual(SymbolicDim lhs, SymbolicDim rhs) {
  SymbolicDim lhsRoot = GetRootSymbolicDim(lhs);
  SymbolicDim rhsRoot = GetRootSymbolicDim(rhs);

  if (lhsRoot != rhsRoot) {
    if (CompareSymbolicDimNames(lhsRoot.getSymName(), rhsRoot.getSymName())) {
      if (!lhsRoot.Merge(rhsRoot)) return false;
      symbolDimUnionSet_[rhsRoot] = lhsRoot;
    } else {
      if (!rhsRoot.Merge(lhsRoot)) return false;
      symbolDimUnionSet_[lhsRoot] = rhsRoot;
    }
  }
  return true;
}

SymbolicDimProduct* SymbolicDimMgr::SymbolicDimProductDivide(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = SimplifySymbolicDimProductPair(lhs, rhs);

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

bool SymbolicDimMgr::IsMultipleOfKnownSymbolicDimProductEqualPair(
    const SymbolicDimProduct& lhs, const SymbolicDimProduct& rhs) {
  for (auto& pairOutter : productEqualityMap_) {
    const SymbolicDimProduct& x = pairOutter.first;
    auto factorX = SymbolicDimProductDivide(lhs, x);
    if (!factorX) continue;
    for (auto& pairInner : pairOutter.second) {
      if (!pairInner.second) continue;
      const SymbolicDimProduct& y = pairInner.first;
      auto factorY = SymbolicDimProductDivide(rhs, y);
      if (!factorY || (*factorX) != (*factorY)) continue;
      return true;
    }
  }

  return false;
}

bool SymbolicDimMgr::UpdateProductEqualityMap() {
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
      std::tie(newX, newY) = SimplifySymbolicDimProductPair(x, y);
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
      if (!IsMultipleOfKnownSymbolicDimProductEqualPair(x, y)) {
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

bool SymbolicDimMgr::IsSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                               const SymbolicDimProduct& rhs) {
  SymbolicDimProduct newLhs, newRhs;
  std::tie(newLhs, newRhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (newLhs == newRhs) return true;
  IR_ENFORCE(UpdateProductEqualityMap(), "Update product equality map failed.");
  return IsMultipleOfKnownSymbolicDimProductEqualPair(newLhs, newRhs);
}

bool SymbolicDimMgr::Save() {
  using Name2SymbolFn = std::function<SymbolicDim(const std::string&)>;
  auto updateAttrs = [&](ArrayAttribute attrs, Name2SymbolFn fn) {
    std::vector<Attribute> newAttrs;
    for (Attribute attr : attrs.AsVector()) {
      auto sym = fn(attr.dyn_cast<StrAttribute>().AsString());
      assert(sym);
      SymbolicDim root = GetRootSymbolicDim(sym);
      Attribute rootSymbol =
          StrAttribute::get(m_->ir_context(), root.getSymName());
      newAttrs.push_back(rootSymbol);
    }
    return ArrayAttribute::get(m_->ir_context(), newAttrs);
  };

  // TODO(liujinnan): update attributes attached in DenseTensorType
  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::getSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::getSymbolicDimAttrName());
    auto symbolicShapeAttr = updateAttrs(attrs, [&](const std::string& name) {
      return symbolTable_.Lookup<SymbolicDim>(name);
    });
    op->set_attribute(SymbolicDim::getSymbolicDimAttrName(), symbolicShapeAttr);
  }
  if (!UpdateProductEqualityMap()) {
    return false;
  }
  std::unordered_set<SymbolicDim, SymDimHasher> usedSymbolicOps;
  std::vector<std::string> usedSymbolNames;
  // TODO(liujinnan): collect uses in value.
  auto collectUsedSymbols = [&](ArrayAttribute attrs) {
    for (Attribute attr : attrs.AsVector()) {
      auto sym = symbolTable_.Lookup<SymbolicDim>(
          attr.dyn_cast<StrAttribute>().AsString());
      assert(sym);
      if (usedSymbolicOps.insert(sym).second)
        usedSymbolNames.push_back(sym.getSymName());
    }
  };
  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::getSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::getSymbolicDimAttrName());
    collectUsedSymbols(attrs);
  }
  auto funcOp = symbolTable_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(funcOp);
  for (auto& p : symbolDimUnionSet_) {
    if (!usedSymbolicOps.count(p.first)) {
      funcOp.block()->erase(*(p.first.operation()));
    }
  }

  std::vector<SymbolicDimProduct> candidates;
  for (auto& outter : productEqualityMap_) {
    if (std::any_of(
            outter.first.symbols.begin(),
            outter.first.symbols.end(),
            [&](SymbolicDim sym) { return usedSymbolicOps.count(sym) == 0; }))
      candidates.push_back(outter.first);
  }

  for (auto& prod : candidates) productEqualityMap_.erase(prod);
  for (auto& outter : productEqualityMap_) {
    std::vector<SymbolicDimProduct> candidates;
    for (auto& inner : outter.second) {
      if (std::any_of(
              inner.first.symbols.begin(),
              inner.first.symbols.end(),
              [&](SymbolicDim sym) { return usedSymbolicOps.count(sym) == 0; }))
        candidates.push_back(outter.first);
    }
    for (auto& prod : candidates) outter.second.erase(prod);
  }

  std::sort(usedSymbolNames.begin(),
            usedSymbolNames.end(),
            [&](const std::string& lhs, const std::string& rhs) {
              return CompareSymbolicDimNames(lhs, rhs);
            });
  int numNonConstDims = 0;
  std::unordered_map<std::string, std::string> nameMapping;
  for (const auto& name : usedSymbolNames) {
    if (name.size() > 0 && name[0] == 'C') {
      nameMapping[name] = name;
    } else {
      nameMapping[name] = ("S" + std::to_string(numNonConstDims++));
    }
  }

  std::unordered_map<std::string, SymbolicDim> name2Symbol;
  for (SymbolicDim op : usedSymbolicOps) {
    auto name = op.getSymName();
    op.updateSymName(nameMapping[name]);
    name2Symbol[name] = op;
  }

  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::getSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::getSymbolicDimAttrName());
    auto symbolicShapeAttr = updateAttrs(
        attrs, [&](const std::string& name) { return name2Symbol[name]; });
    op->set_attribute(SymbolicDim::getSymbolicDimAttrName(), symbolicShapeAttr);
  }

  // TODO(liujinnan): update attributes attached to values.

  return SaveShapeConstraintGraph();
}

bool SymbolicDimMgr::SaveShapeConstraintGraph() {
  auto funcOp = symbolTable_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(funcOp);
  auto op_it = funcOp.block()->rbegin();
  while (op_it != funcOp.block()->rend()) {
    if (((*op_it)->isa<dialect::SymbolicDim>()) ||
        ((*op_it)->isa<dialect::TieShapeOp>()))
      op_it++;
    else
      op_it = decltype(op_it)(funcOp.block()->erase(*(*op_it)));
  }

  Builder builder = Builder(m_->ir_context(), funcOp.block());
  auto build_operands = [&](const SymbolicDimProduct& prod) {
    std::vector<Value> values;

    if (prod.factor != 1) {
      values.push_back(
          builder
              .Build<ConstantOp>(
                  Int32Attribute::get(m_->ir_context(), prod.factor),
                  Int32Type::get(m_->ir_context()))
              ->result(0));
    }
    for (SymbolicDim sym : prod.symbols) {
      values.push_back(builder.Build<dialect::DimOp>(sym.getSymName()).out());
    }
    return values;
  };
  std::vector<SymbolicDimProduct> sortedProductVec;
  for (auto& p : productEqualityMap_) sortedProductVec.push_back(p.first);
  std::sort(sortedProductVec.begin(),
            sortedProductVec.end(),
            CompareSymbolicDimProduct);
  for (auto& x : sortedProductVec) {
    for (auto& y : sortedProductVec) {
      if (!CompareSymbolicDimProduct(x, y)) continue;
      if (!productEqualityMap_[x][y]) continue;
      auto lhsOperands = build_operands(x);
      auto rhsOperands = build_operands(y);
      builder.Build<dialect::TieProductEqualOp>(lhsOperands, rhsOperands);
    }
  }
  return true;
}

bool ShapeAnalysis::IsSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto lhsTy = lhs.type().dyn_cast_interface<ShapedTypeInterface>();
  auto rhsTy = rhs.type().dyn_cast_interface<ShapedTypeInterface>();

  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

  return IsProductEqual(lhs, 0, lhsTy.getRank(), rhs, 0, rhsTy.getRank());
}

bool ShapeAnalysis::IsProductEqual(
    Value lhs, int lhsFrom, int lhsTo, Value rhs, int rhsFrom, int rhsTo) {
  std::vector<int> lhsDimIdxs, rhsDimIdxs;
  lhsDimIdxs.reserve(lhsTo - lhsFrom);
  rhsDimIdxs.reserve(rhsTo - rhsFrom);
  for (int i = lhsFrom; i < lhsTo; ++i) lhsDimIdxs.push_back(i);
  for (int i = rhsFrom; i < rhsTo; ++i) rhsDimIdxs.push_back(i);

  return IsProductEqual(lhs, lhsDimIdxs, rhs, rhsDimIdxs);
}

SymbolicDimShapeAnalysis::SymbolicDimShapeAnalysis(ModuleOp m)
    : m_(m), mgr_(m) {
  mgr_.Load();
  for (auto op : *(m_.block())) {
    auto tieShapeOp = op->dyn_cast<dialect::TieShapeOp>();
    if (!tieShapeOp) continue;
    Value result = tieShapeOp.value();
    auto& symbols = value2SymDims_[result];
    auto attrs =
        tieShapeOp
            .attribute<ArrayAttribute>(SymbolicDim::getSymbolicDimAttrName())
            .AsVector();
    for (const auto& attr : attrs) {
      auto symOp = mgr_.symbolTable().Lookup<SymbolicDim>(
          attr.dyn_cast<StrAttribute>().AsString());
      if (!symOp) continue;
      symbols.push_back(symOp);
    }
  }
}

SymbolicDimShapeAnalysis::~SymbolicDimShapeAnalysis() { mgr_.Save(); }

bool SymbolicDimShapeAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhsTy = lhs.type().dyn_cast_interface<ShapedTypeInterface>();
  auto rhsTy = rhs.type().dyn_cast_interface<ShapedTypeInterface>();

  if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

  if (lhsTy.hasStaticShape() && rhsTy.hasStaticShape()) {
    return vectorize(lhsTy.getShape()) == vectorize(rhsTy.getShape());
  }

  auto lhsIt = value2SymDims_.find(lhs);
  auto rhsIt = value2SymDims_.find(rhs);

  if (lhsIt == value2SymDims_.end() || rhsIt == value2SymDims_.end() ||
      lhsIt->second.size() != rhsIt->second.size())
    return false;

  std::vector<SymbolicDim> lhsSyms;
  std::vector<SymbolicDim> rhsSyms;
  for (auto sym : lhsIt->second) {
    lhsSyms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  for (auto sym : rhsIt->second) {
    rhsSyms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  return lhsSyms == rhsSyms;
}

bool SymbolicDimShapeAnalysis::IsProductEqual(Value lhs,
                                              std::vector<int> lhsDimIdxs,
                                              Value rhs,
                                              std::vector<int> rhsDimIdxs) {
  SymbolicDimProduct lhsProd;
  SymbolicDimProduct rhsProd;

  auto buildSymbolicDimProduct =
      [&](SymbolicDimProduct& prod, Value value, std::vector<int> dimIdxs) {
        auto ty = value.type().dyn_cast_interface<ShapedTypeInterface>();
        auto it = value2SymDims_.find(value);
        if (!ty || !ty.hasRank()) return false;
        for (int idx : dimIdxs) {
          if (ty.getShape()[idx] == ShapedTypeInterface::kDynamic) {
            if (it == value2SymDims_.end() ||
                static_cast<int>(it->second.size()) <= idx)
              return false;
            prod.symbols.push_back(it->second[idx]);
          } else {
            prod.factor *= ty.getShape()[idx];
          }
        }
        return true;
      };

  if (!buildSymbolicDimProduct(lhsProd, lhs, lhsDimIdxs) ||
      !buildSymbolicDimProduct(rhsProd, rhs, rhsDimIdxs)) {
    return false;
  }

  return mgr_.IsSymbolicDimProductEqual(lhsProd, rhsProd);
}

ShapeComputationIRAnalysis::ShapeComputationIRAnalysis(ModuleOp m,
                                                       SymbolicDimMgr& mgr)
    : m_(m), mgr_(mgr) {}

bool ShapeComputationIRAnalysis::Run() {
  // Make sure only run once.
  if (initialized_) return false;
  initialized_ = true;
  auto buildShapeFunc =
      std::bind(&ShapeComputationIRAnalysis::BuildShapeOnOperation,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), buildShapeFunc)) return false;
  auto applyOpConstraintFunc =
      std::bind(&ShapeComputationIRAnalysis::ApplyOpConstraint,
                this,
                std::placeholders::_1);
  if (!RunOnRegion(&(m_->region(0)), applyOpConstraintFunc)) return false;
  return true;
}

bool ShapeComputationIRAnalysis::RunOnRegion(Region* region, func fn) {
  for (Block* block : *region) {
    if (!RunOnBlock(block, fn)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::RunOnBlock(Block* block, func fn) {
  // TODO(liujinnan): mapping block arguments

  std::vector<Operation*> op_list;
  for (Operation* op : *block) op_list.push_back(op);
  for (Operation* op : op_list) {
    if (!RunOnOperation(op, fn)) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::RunOnOperation(Operation* op, func fn) {
  for (size_t i = 0; i < op->num_regions(); ++i) {
    if (!RunOnRegion(&(op->region(i)), fn)) return false;
  }
  return fn(op);
}

bool ShapeComputationIRAnalysis::BuildShapeOnOperation(Operation* op) {
  if (op->isa<dialect::FuncOp>()) return true;
  if (op->isa<dialect::TieShapeOp>()) {
    Value value = op->operand_source(0);
    std::vector<SymbolicDim> symbols;
    if (op->HasAttribute(SymbolicDim::getSymbolicDimAttrName())) {
      auto attrs =
          op->attribute<ArrayAttribute>(SymbolicDim::getSymbolicDimAttrName())
              .AsVector();
      for (Attribute attr : attrs) {
        auto sym = mgr_.symbolTable().Lookup<SymbolicDim>(
            attr.dyn_cast<StrAttribute>().AsString());
        assert(sym);
        SymbolicDim root = mgr_.GetRootSymbolicDim(sym);
        symbols.push_back(root);
      }
    } else {
      symbols = mgr_.CreateSymbolicDimsForRankedValue(value);
      std::vector<Attribute> attrs;
      for (SymbolicDim sym : symbols) {
        Attribute rootSymbol =
            StrAttribute::get(m_->ir_context(), sym.getSymName());
        attrs.push_back(rootSymbol);
      }
      op->set_attribute(SymbolicDim::getSymbolicDimAttrName(),
                        ArrayAttribute::get(m_->ir_context(), attrs));
    }
    rankedTensor2SymDims_[value] = std::move(symbols);
    return true;
  }
  for (size_t i = 0; i < op->num_results(); ++i) {
    if (!BuildShapeOnValue(op->result(i))) return false;
  }
  return true;
}

bool ShapeComputationIRAnalysis::BuildShapeOnValue(Value value) {
  Type ty = value.type();
  if (IsIntOrIndex(ty)) {
    SymbolicDim sym = mgr_.NewSymbolicDim();
    value2SymDim_[value] = sym;
  } else if (IsCandidateShapeTensorType(ty)) {
    auto shapedTy = ty.dyn_cast_interface<ShapedTypeInterface>();
    std::vector<SymbolicDim> symbols;
    for (size_t i = 0, d = shapedTy.getShape()[0]; i < d; ++i)
      symbols.push_back(mgr_.NewSymbolicDim());
    shapeTensor2SymDims_[value] = std::move(symbols);
  }
  return true;
}

bool ShapeComputationIRAnalysis::ApplyOpConstraint(Operation* op) {
  IR_ENFORCE(ApplyIndexOpConstraint(op),
             "Fail to apply constraint for index op");
  IR_ENFORCE(ApplyTieShapeOpConstraint(op),
             "Fail to apply constraint for tie_shape op");

  // TODO(zhangbo63): add more constraints
  return true;
}

bool ShapeComputationIRAnalysis::ApplyIndexOpConstraint(Operation* op) {
  if (op->num_results() == 0) return true;

  Type ty = op->result(0).type();
  if (!IsIntOrIndex(ty)) return true;

  if (auto dimOp = op->dyn_cast<dialect::TensorDimOp>()) {
    int64_t dimIndex = dimOp.index()
                           .dyn_cast<OpResult>()
                           .owner()
                           ->attribute<Int64Attribute>("value")
                           .data();
    value2SymDim_[dimOp.out()].updateKnownNonNegative(true);
    if (!mgr_.MapSymbolicDimEqual(
            value2SymDim_[dimOp.out()],
            rankedTensor2SymDims_[dimOp.source()][dimIndex])) {
      return false;
    }

  } else if (auto constOp = op->dyn_cast<ConstantOp>()) {
    int64_t val = constOp.value().dyn_cast<Int64Attribute>().data();
    if (!mgr_.MapSymbolicDimEqual(value2SymDim_[op->result(0)],
                                  mgr_.NewConstantSymbolicDim(val))) {
      return false;
    }
  }
  // TODO(zhangbo63): add support for reifyInferShape. (e.g. mul/add)
  return true;
}

bool ShapeComputationIRAnalysis::ApplyTieShapeOpConstraint(Operation* op) {
  if (auto tieShape = op->dyn_cast<dialect::TieShapeOp>()) {
    auto& value = rankedTensor2SymDims_[op->operand_source(0)];
    for (size_t idx = 0; idx < tieShape.dims().size(); ++idx) {
      if (!mgr_.MapSymbolicDimEqual(value2SymDim_[tieShape.dims()[idx]],
                                    value[idx]))
        return false;
      mgr_.GetRootSymbolicDim(value[idx]).updateKnownNonNegative(true);
    }
  }
  return true;
}

bool IsIntOrIndex(Type type) {
  return type.isa<IndexType>() || type.isa<Int8Type>() ||
         type.isa<UInt8Type>() || type.isa<Int16Type>() ||
         type.isa<Int32Type>() || type.isa<Int64Type>();
}

bool IsCandidateShapeTensorType(Type ty) {
  if (auto tensorTy = ty.dyn_cast<paddle::dialect::DenseTensorType>()) {
    auto shapedTy = tensorTy.dyn_cast_interface<ShapedTypeInterface>();
    return (shapedTy.getRank() == 1 && shapedTy.hasStaticShape() &&
            IsIntOrIndex(shapedTy.getElementType()) &&
            shapedTy.getShape()[0] < 32);
  }
  return false;
}

}  // namespace pir
