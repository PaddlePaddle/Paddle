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

bool ShapeAnalysis::IsSameNumElements(Value lhs, Value rhs) {
  if (lhs == rhs) return true;
  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  return IsProductEqual(lhs,
                        0,
                        static_cast<int>(lhs_type.GetRank()),
                        rhs,
                        0,
                        static_cast<int>(rhs_type.GetRank()));
}

bool ShapeAnalysis::IsProductEqual(
    Value lhs, int lhs_from, int lhs_to, Value rhs, int rhs_from, int rhs_to) {
  std::vector<int> lhs_dim_idxs, rhs_dim_idxs;

  lhs_dim_idxs.reserve(lhs_to - lhs_from);
  rhs_dim_idxs.reserve(rhs_to - rhs_from);

  for (int i = lhs_from; i < lhs_to; ++i) lhs_dim_idxs.push_back(i);
  for (int i = rhs_from; i < rhs_to; ++i) rhs_dim_idxs.push_back(i);

  return IsProductEqual(lhs, lhs_dim_idxs, rhs, rhs_dim_idxs);
}

ShapeConstraintIRAnalysis::ShapeConstraintIRAnalysis(ModuleOp m)
    : m_(m), mgr_(m) {
  mgr_.Load();
  for (auto op : *(m_.block())) {
    auto tie_shape_op = op->dyn_cast<dialect::TieShapeOp>();
    if (!tie_shape_op) continue;
    Value result = tie_shape_op.value();
    auto& symbols = value_to_sym_dims_[result];
    auto attrs =
        tie_shape_op
            .attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName())
            .AsVector();
    for (const auto& attr : attrs) {
      auto sym_op = mgr_.symbolTable().Lookup<SymbolicDim>(
          attr.dyn_cast<StrAttribute>().AsString());
      if (!sym_op) continue;
      symbols.push_back(sym_op);
    }
  }
}

ShapeConstraintIRAnalysis::~ShapeConstraintIRAnalysis() { mgr_.Save(); }

bool ShapeConstraintIRAnalysis::IsShapeEqual(Value lhs, Value rhs) {
  if (lhs == rhs) return true;

  auto lhs_type = lhs.type().dyn_cast<ShapedTypeInterface>();
  auto rhs_type = rhs.type().dyn_cast<ShapedTypeInterface>();

  if (!lhs_type || !rhs_type || !lhs_type.HasRank() || !rhs_type.HasRank())
    return false;

  if (lhs_type.HasStaticShape() && rhs_type.HasStaticShape()) {
    return vectorize(lhs_type.GetShape()) == vectorize(rhs_type.GetShape());
  }

  auto lhs_it = value_to_sym_dims_.find(lhs);
  auto rhs_it = value_to_sym_dims_.find(rhs);

  if (lhs_it == value_to_sym_dims_.end() ||
      rhs_it == value_to_sym_dims_.end() ||
      lhs_it->second.size() != rhs_it->second.size())
    return false;

  std::vector<SymbolicDim> lhs_syms;
  std::vector<SymbolicDim> rhs_syms;
  for (auto sym : lhs_it->second) {
    lhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  for (auto sym : rhs_it->second) {
    rhs_syms.push_back(mgr_.GetRootSymbolicDim(sym));
  }
  return lhs_syms == rhs_syms;
}

bool ShapeConstraintIRAnalysis::IsProductEqual(Value lhs,
                                               std::vector<int> lhs_dim_idxs,
                                               Value rhs,
                                               std::vector<int> rhs_dim_idxs) {
  SymbolicDimProduct lhs_prod;
  SymbolicDimProduct rhs_prod;

  auto build_symbolic_dim_product =
      [&](SymbolicDimProduct& prod, Value value, std::vector<int> dim_idxs) {
        auto type = value.type().dyn_cast<ShapedTypeInterface>();
        auto it = value_to_sym_dims_.find(value);
        if (!type || !type.HasRank()) return false;
        for (int idx : dim_idxs) {
          if (type.GetShape()[idx] == ShapedTypeInterface::kDynamic) {
            if (it == value_to_sym_dims_.end() ||
                static_cast<int>(it->second.size()) <= idx)
              return false;
            prod.symbols.push_back(it->second[idx]);
          } else {
            prod.factor *= type.GetShape()[idx];
          }
        }
        return true;
      };

  if (!build_symbolic_dim_product(lhs_prod, lhs, lhs_dim_idxs) ||
      !build_symbolic_dim_product(rhs_prod, rhs, rhs_dim_idxs)) {
    return false;
  }

  return mgr_.IsSymbolicDimProductEqual(lhs_prod, rhs_prod);
}

// Gives a consistent order of a list op SymbolicDim Ops
bool CompareSymbolicDimNames(const std::string& lhs, const std::string& rhs) {
  // S -> unknown dimension size at compile time
  // C -> constant dimension size at compile time
  if (lhs.size() < 1 || (lhs[0] != 'S' && lhs[0] != 'C')) return lhs < rhs;
  if (rhs.size() < 1 || (rhs[0] != 'S' && rhs[0] != 'C')) return lhs < rhs;
  int64_t lhs_idx = 0, rhs_idx = 0;
  try {
    lhs_idx = stol(lhs.substr(1));
    rhs_idx = stol(rhs.substr(1));
  } catch (const std::exception& e) {
    IR_THROW("Invalid symbolic name");
  }
  return (lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhs_idx < rhs_idx);
}

// Gives a consistent order of a list op SymbolicDimProducts
bool CompareSymbolicDimProduct(SymbolicDimProduct& lhs,    // NOLINT
                               SymbolicDimProduct& rhs) {  // NOLINT
  if (lhs.symbols.size() < rhs.symbols.size()) return true;
  if (lhs.symbols.size() == rhs.symbols.size()) {
    for (size_t idx = 0; idx < lhs.symbols.size(); ++idx) {
      const std::string lhs_name = lhs.symbols[idx].GetSymName();
      const std::string rhs_name = rhs.symbols[idx].GetSymName();
      if (CompareSymbolicDimNames(lhs_name, rhs_name)) return true;
      if (lhs_name != rhs_name) return false;
    }
  }
  return false;
}

bool SymbolicDimMgr::Load() {
  auto func_op = symbol_table_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(func_op);
  for (auto op_ : *(func_op.block())) {
    symbol_table_.insert(op_);
    if (SymbolicDim op = op_->dyn_cast<SymbolicDim>()) {
      symbolDimUnionSet_[op] = op;
      symbolNameSet_.insert(op.GetSymName());
    }
  }
  return LoadShapeConstraintGraph();
}

bool SymbolicDimMgr::LoadShapeConstraintGraph() {
  // TODO(liujinnan): add more constraint function. currently, only support
  // tie_product_equal.
  auto constraint_vec =
      symbol_table_.Lookup<dialect::TieProductEqualOp>("tie_product_equal");

  if (!constraint_vec.size()) return true;

  auto build_sym_product = [&](std::vector<Value> range,
                               SymbolicDimProduct& product) {
    for (Value v : range) {
      auto definingOp = v.dyn_cast<OpResult>().owner();
      if (auto constOp = definingOp->dyn_cast<ConstantOp>()) {
        product.factor *= constOp.value().dyn_cast<Int32Attribute>().data();
        continue;
      } else if (auto dimOp = definingOp->dyn_cast<dialect::DimOp>()) {
        auto sym = symbol_table_.Lookup<SymbolicDim>(dimOp.getName());
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

bool SymbolicDimMgr::MapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                                const SymbolicDimProduct& rhs) {
  SymbolicDimProduct new_lhs, new_rhs;
  std::tie(new_lhs, new_rhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (new_lhs == new_rhs) return true;

  if (new_lhs.factor == new_rhs.factor && new_lhs.symbols.size() == 1 &&
      new_rhs.symbols.size() == 1) {
    return MapSymbolicDimEqual(new_lhs.symbols[0], new_rhs.symbols[0]);
  } else if (new_lhs.symbols.size() == 0 && new_rhs.symbols.size() == 1 &&
             new_rhs.factor == 1) {
    return MapSymbolicDimEqual(NewConstantSymbolicDim(new_lhs.factor),
                               new_rhs.symbols[0]);
  } else if (new_rhs.symbols.size() == 0 && new_lhs.symbols.size() == 1 &&
             new_lhs.factor == 1) {
    return MapSymbolicDimEqual(NewConstantSymbolicDim(new_rhs.factor),
                               new_lhs.symbols[0]);
  }

  productEqualityMap_[new_lhs][new_rhs] =
      productEqualityMap_[new_rhs][new_lhs] = true;

  productEqualityMapUpdated_ = false;
  return true;
}

std::pair<SymbolicDimProduct, SymbolicDimProduct>
SymbolicDimMgr::SimplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y) {
  auto lhs = SimplifySymbolicDimProduct(x);
  auto rhs = SimplifySymbolicDimProduct(y);

  SymbolicDimProduct new_lhs, new_rhs;
  int64_t gcd_factor = std::gcd(std::abs(lhs.factor), std::abs(rhs.factor));
  if (!gcd_factor)
    return std::make_pair(std::move(new_lhs), std::move(new_rhs));
  if (std::abs(lhs.factor) < std::abs(rhs.factor)) {
    if (lhs.factor < 0) gcd_factor = -gcd_factor;
  } else {
    if (rhs.factor < 0) gcd_factor = -gcd_factor;
  }

  new_lhs.factor = lhs.factor / gcd_factor;
  new_rhs.factor = rhs.factor / gcd_factor;

  std::unordered_map<SymbolicDim, int, SymDimHasher> lhs_symbol_map;
  std::unordered_map<SymbolicDim, int, SymDimHasher> rhs_symbol_map;
  for (SymbolicDim op : lhs.symbols) ++lhs_symbol_map[op];
  for (SymbolicDim op : rhs.symbols) ++rhs_symbol_map[op];

  for (SymbolicDim op : lhs.symbols) {
    auto it = rhs_symbol_map.find(op);
    if (it != rhs_symbol_map.end() && op.GetKnownNonSizeZero()) {
      if (--it->second == 0) rhs_symbol_map.erase(it);
      continue;
    }
    new_lhs.symbols.push_back(op);
  }

  for (SymbolicDim op : rhs.symbols) {
    auto it = lhs_symbol_map.find(op);
    if (it != lhs_symbol_map.end() && op.GetKnownNonSizeZero()) {
      if (--it->second == 0) lhs_symbol_map.erase(it);
      continue;
    }
    new_rhs.symbols.push_back(op);
  }

  if (!new_lhs.factor) new_lhs.symbols.clear();
  if (!new_rhs.factor) new_rhs.symbols.clear();

  return std::make_pair(std::move(new_lhs), std::move(new_rhs));
}

SymbolicDimProduct SymbolicDimMgr::SimplifySymbolicDimProduct(
    const SymbolicDimProduct& x) {
  std::vector<SymbolicDim> copied;
  copied.reserve(x.symbols.size());
  for (SymbolicDim op : x.symbols) copied.push_back(GetRootSymbolicDim(op));

  sort(copied.begin(), copied.end(), [&](SymbolicDim lhs, SymbolicDim rhs) {
    return CompareSymbolicDimNames(lhs.GetSymName(), rhs.GetSymName());
  });
  SymbolicDimProduct newX;
  newX.factor = x.factor;
  for (SymbolicDim op : copied) {
    if (!op.IsDynamic()) {
      newX.factor *= op.GetDimSize();
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
      symbol_table_ = SymbolTable(op);
      return;
    }
  }
  Builder builder = Builder(m_.ir_context(), m_.block(), m_.block()->begin());
  dialect::FuncOp func = builder.Build<dialect::FuncOp>();
  symbol_table_ = SymbolTable(func);
}

SymbolicDim SymbolicDimMgr::NewSymbolicDim(const std::string& name) {
  auto func_op = symbol_table_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(func_op);
  Builder builder = Builder(m_.ir_context(), func_op.block());
  // default settting dim != 0
  dialect::SymbolicDim symbol =
      builder.Build<dialect::SymbolicDim>(name.empty() ? GetNextName() : name,
                                          ShapedTypeInterface::kDynamic,
                                          false,
                                          false,
                                          false,
                                          true);
  symbolDimUnionSet_[symbol] = symbol;
  symbol_table_.insert(symbol);
  return symbol;
}

SymbolicDim SymbolicDimMgr::NewConstantSymbolicDim(int64_t val) {
  auto it = constantSymbolicDimMap_.find(val);
  if (it == constantSymbolicDimMap_.end()) {
    auto name = "C" + std::to_string(val);
    it = constantSymbolicDimMap_
             .insert(std::make_pair(val, NewSymbolicDim(name)))
             .first;
    it->second.SetDimSize(val);
    if (val == -1) it->second.UpdateKnownNegativeOne(true);
    if (val >= 0) it->second.UpdateKnownNonNegative(true);
    if (val != 1) it->second.UpdateKnownNonSizeOne(true);
    if (val != 0) it->second.UpdateKnownNonSizeZero(true);
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
    if (CompareSymbolicDimNames(lhsRoot.GetSymName(), rhsRoot.GetSymName())) {
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
  SymbolicDimProduct new_lhs, new_rhs;
  std::tie(new_lhs, new_rhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  if (new_lhs.factor == 0 || new_rhs.factor == 0) return nullptr;
  if (new_lhs.factor % new_rhs.factor != 0) return nullptr;
  if (new_lhs.symbols.size() < new_rhs.symbols.size()) return nullptr;

  SymbolicDimProduct* result = new SymbolicDimProduct();
  result->factor = new_lhs.factor / new_rhs.factor;

  std::unordered_map<SymbolicDim, int, SymDimHasher> sym_proc_map;
  for (SymbolicDim sym : new_rhs.symbols) ++sym_proc_map[sym];

  for (SymbolicDim sym : new_lhs.symbols) {
    auto it = sym_proc_map.find(sym);
    if (it == sym_proc_map.end()) {
      result->symbols.push_back(sym);
      continue;
    }
    if (--it->second == 0) {
      sym_proc_map.erase(it);
      continue;
    }
  }

  if (!sym_proc_map.empty()) return nullptr;
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
  SymbolicDimProduct new_lhs, new_rhs;
  std::tie(new_lhs, new_rhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // early return for identity case.
  if (new_lhs == new_rhs) return true;
  IR_ENFORCE(UpdateProductEqualityMap(), "Update product equality map failed.");
  return IsMultipleOfKnownSymbolicDimProductEqualPair(new_lhs, new_rhs);
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
          StrAttribute::get(m_->ir_context(), root.GetSymName());
      newAttrs.push_back(rootSymbol);
    }
    return ArrayAttribute::get(m_->ir_context(), newAttrs);
  };

  // TODO(liujinnan): update attributes attached in DenseTensorType
  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::GetSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName());
    auto symbolicShapeAttr = updateAttrs(attrs, [&](const std::string& name) {
      return symbol_table_.Lookup<SymbolicDim>(name);
    });
    op->set_attribute(SymbolicDim::GetSymbolicDimAttrName(), symbolicShapeAttr);
  }
  if (!UpdateProductEqualityMap()) {
    return false;
  }
  std::unordered_set<SymbolicDim, SymDimHasher> usedSymbolicOps;
  std::vector<std::string> usedSymbolNames;
  // TODO(liujinnan): collect uses in value.
  auto collectUsedSymbols = [&](ArrayAttribute attrs) {
    for (Attribute attr : attrs.AsVector()) {
      auto sym = symbol_table_.Lookup<SymbolicDim>(
          attr.dyn_cast<StrAttribute>().AsString());
      assert(sym);
      if (usedSymbolicOps.insert(sym).second)
        usedSymbolNames.push_back(sym.GetSymName());
    }
  };
  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::GetSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName());
    collectUsedSymbols(attrs);
  }
  auto func_op = symbol_table_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(func_op);
  for (auto& p : symbolDimUnionSet_) {
    if (!usedSymbolicOps.count(p.first)) {
      func_op.block()->erase(*(p.first.operation()));
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
    auto name = op.GetSymName();
    op.SetSymName(nameMapping[name]);
    name2Symbol[name] = op;
  }

  for (auto op : *(m_.block())) {
    if (!op->HasAttribute(SymbolicDim::GetSymbolicDimAttrName())) continue;
    auto attrs =
        op->attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName());
    auto symbolicShapeAttr = updateAttrs(
        attrs, [&](const std::string& name) { return name2Symbol[name]; });
    op->set_attribute(SymbolicDim::GetSymbolicDimAttrName(), symbolicShapeAttr);
  }

  // TODO(liujinnan): update attributes attached to values.

  return SaveShapeConstraintGraph();
}

bool SymbolicDimMgr::SaveShapeConstraintGraph() {
  auto func_op = symbol_table_.getOp()->dyn_cast<dialect::FuncOp>();
  assert(func_op);
  auto op_it = func_op.block()->rbegin();
  while (op_it != func_op.block()->rend()) {
    if (((*op_it)->isa<dialect::SymbolicDim>()) ||
        ((*op_it)->isa<dialect::TieShapeOp>()))
      op_it++;
    else
      op_it = decltype(op_it)(func_op.block()->erase(*(*op_it)));
  }

  Builder builder = Builder(m_->ir_context(), func_op.block());
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
      values.push_back(builder.Build<dialect::DimOp>(sym.GetSymName()).out());
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
    if (op->HasAttribute(SymbolicDim::GetSymbolicDimAttrName())) {
      auto attrs =
          op->attribute<ArrayAttribute>(SymbolicDim::GetSymbolicDimAttrName())
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
            StrAttribute::get(m_->ir_context(), sym.GetSymName());
        attrs.push_back(rootSymbol);
      }
      op->set_attribute(SymbolicDim::GetSymbolicDimAttrName(),
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
  Type type = value.type();
  if (IsIntOrIndex(type)) {
    SymbolicDim sym = mgr_.NewSymbolicDim();
    value2SymDim_[value] = sym;
  } else if (IsCandidateShapeTensorType(type)) {
    auto shapedTy = type.dyn_cast<ShapedTypeInterface>();
    std::vector<SymbolicDim> symbols;
    for (size_t i = 0, d = shapedTy.GetShape()[0]; i < d; ++i)
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

  Type type = op->result(0).type();
  if (!IsIntOrIndex(type)) return true;

  if (auto dimOp = op->dyn_cast<dialect::TensorDimOp>()) {
    int64_t dimIndex = dimOp.index()
                           .dyn_cast<OpResult>()
                           .owner()
                           ->attribute<Int64Attribute>("value")
                           .data();
    value2SymDim_[dimOp.out()].UpdateKnownNonNegative(true);
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
      mgr_.GetRootSymbolicDim(value[idx]).UpdateKnownNonNegative(true);
    }
  }
  return true;
}

bool IsIntOrIndex(Type type) {
  return type.isa<IndexType>() || type.isa<Int8Type>() ||
         type.isa<UInt8Type>() || type.isa<Int16Type>() ||
         type.isa<Int32Type>() || type.isa<Int64Type>();
}

bool IsCandidateShapeTensorType(Type type) {
  if (auto tensorTy = type.dyn_cast<paddle::dialect::DenseTensorType>()) {
    auto shapedTy = tensorTy.dyn_cast<ShapedTypeInterface>();
    return (shapedTy.GetRank() == 1 && shapedTy.HasStaticShape() &&
            IsIntOrIndex(shapedTy.GetElementType()) &&
            shapedTy.GetShape()[0] < 32);
  }
  return false;
}

}  // namespace pir
