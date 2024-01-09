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

#include "paddle/pir/dialect/shape/utils/shape_optimization_utils.h"
#include "paddle/pir/core/builtin_type.h"
#include "paddle/pir/dialect/shape/utils/symbol_table.h"

namespace pir {

bool CompareSymbolicDimNames(const std::string& lhs, const std::string& rhs) {
  // S -> Symbol   : unknown  dimension size at compile time
  // C -> Constant : constant dimension size at compile time
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

SymbolicDimMgr::SymbolicDimMgr(ModuleOp m) : m_(m) {}

bool SymbolicDimMgr::MapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                                const SymbolicDimProduct& rhs) {
  SymbolicDimProduct new_lhs, new_rhs;
  std::tie(new_lhs, new_rhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // Return true for identity case.
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

  product_equality_map_[new_lhs][new_rhs] =
      product_equality_map_[new_rhs][new_lhs] = true;

  product_equality_map_updated_ = false;
  return true;
}

SymbolicDimProduct SymbolicDimMgr::SimplifySymbolicDimProduct(
    const SymbolicDimProduct& x) {
  std::vector<SymbolicDimOp> copied;
  copied.reserve(x.symbols.size());
  for (SymbolicDimOp op : x.symbols) copied.push_back(GetRootSymbolicDim(op));

  std::sort(
      copied.begin(), copied.end(), [&](SymbolicDimOp lhs, SymbolicDimOp rhs) {
        return CompareSymbolicDimNames(lhs.GetSymName(), rhs.GetSymName());
      });
  SymbolicDimProduct new_x;
  new_x.factor = x.factor;
  for (SymbolicDimOp op : copied) {
    if (!op.IsDynamic()) {
      new_x.factor *= op.GetDimSize();
    } else {
      new_x.symbols.push_back(op);
    }
  }
  return new_x;
}

std::pair<SymbolicDimProduct, SymbolicDimProduct>
SymbolicDimMgr::SimplifySymbolicDimProductPair(const SymbolicDimProduct& x,
                                               const SymbolicDimProduct& y) {
  // First do some basic clean up (e.g. folding const symbolic dim op into the
  // fator field)
  auto lhs = SimplifySymbolicDimProduct(x);
  auto rhs = SimplifySymbolicDimProduct(y);

  SymbolicDimProduct new_lhs, new_rhs;
  int64_t gcd_factor = std::gcd(std::abs(lhs.factor), std::abs(rhs.factor));

  // 0 * lhs_symbols = 0 * rhs_symbols, no more information.
  // Just return empty new_lhs & new_rhs
  if (!gcd_factor)
    return std::make_pair(std::move(new_lhs), std::move(new_rhs));

  // Canonicalization factor form: always let the smaller factor being positive
  // number.
  if (std::abs(lhs.factor) < std::abs(rhs.factor)) {
    if (lhs.factor < 0) gcd_factor = -gcd_factor;
  } else {
    if (rhs.factor < 0) gcd_factor = -gcd_factor;
  }

  new_lhs.factor = lhs.factor / gcd_factor;
  new_rhs.factor = rhs.factor / gcd_factor;

  std::unordered_map<SymbolicDimOp, int, SymDimHasher> lhs_symbol_map;
  std::unordered_map<SymbolicDimOp, int, SymDimHasher> rhs_symbol_map;

  for (SymbolicDimOp op : lhs.symbols) ++lhs_symbol_map[op];
  for (SymbolicDimOp op : rhs.symbols) ++rhs_symbol_map[op];

  for (SymbolicDimOp op : lhs.symbols) {
    auto it = rhs_symbol_map.find(op);
    if (it != rhs_symbol_map.end() && op.GetKnownNonSizeZero()) {
      if (--it->second == 0) rhs_symbol_map.erase(it);
      continue;
    }
    new_lhs.symbols.push_back(op);
  }

  for (SymbolicDimOp op : rhs.symbols) {
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

const std::string SymbolicDimMgr::GetNextName() {
  std::string name;
  do {
    name = "S" + std::to_string(next_symbolic_idx_++);
  } while (!symbol_name_set_.insert(name).second);
  return name;
}

SymbolicDimOp SymbolicDimMgr::NewSymbolicDim(const std::string& name) {
  auto func_op = symbol_table_.getOp()->dyn_cast<shape::FuncOp>();
  IR_ENFORCE(func_op);
  Builder builder = Builder(m_.ir_context(), func_op.block());
  // default settting dim != 0
  SymbolicDimOp symbol =
      builder.Build<SymbolicDimOp>(name.empty() ? GetNextName() : name,
                                   ShapedTypeInterface::kDynamic,
                                   false,
                                   false,
                                   false,
                                   true);
  symbol_dim_union_set_[symbol] = symbol;
  symbol_table_.insert(symbol);
  return symbol;
}

SymbolicDimOp SymbolicDimMgr::NewConstantSymbolicDim(int64_t val) {
  auto it = constant_symbolic_dim_map_.find(val);
  if (it == constant_symbolic_dim_map_.end()) {
    auto name = "C" + std::to_string(val);
    it = constant_symbolic_dim_map_
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

std::vector<SymbolicDimOp> SymbolicDimMgr::CreateSymbolicDimsForRankedValue(
    Value value) {
  std::vector<SymbolicDimOp> symbols;
  auto dims = value.type().dyn_cast<pir::DenseTensorType>().dims();
  for (int idx = 0; idx < dims.size(); ++idx) {
    symbols.push_back(
        (dims[idx] == ShapedTypeInterface::kDynamic || dims[idx] == -1)
            ? NewSymbolicDim()
            : NewConstantSymbolicDim(dims[idx]));
  }
  return symbols;
}

SymbolicDimOp SymbolicDimMgr::GetRootSymbolicDim(SymbolicDimOp symbol) {
  SymbolicDimOp current = symbol;
  std::vector<SymbolicDimOp> path;
  while (symbol_dim_union_set_[current] != current) {
    path.push_back(current);
    current = symbol_dim_union_set_[current];
  }
  for (SymbolicDimOp sym : path) symbol_dim_union_set_[sym] = current;
  return current;
}

bool SymbolicDimMgr::IsSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs) {
  SymbolicDimOp lhs_root = GetRootSymbolicDim(lhs);
  SymbolicDimOp rhs_root = GetRootSymbolicDim(rhs);
  return lhs_root == rhs_root;
}

bool SymbolicDimMgr::MapSymbolicDimEqual(SymbolicDimOp lhs, SymbolicDimOp rhs) {
  SymbolicDimOp lhs_root = GetRootSymbolicDim(lhs);
  SymbolicDimOp rhs_root = GetRootSymbolicDim(rhs);

  if (lhs_root != rhs_root) {
    if (CompareSymbolicDimNames(lhs_root.GetSymName(), rhs_root.GetSymName())) {
      if (!lhs_root.Merge(rhs_root)) return false;
      symbol_dim_union_set_[rhs_root] = lhs_root;
    } else {
      if (!rhs_root.Merge(lhs_root)) return false;
      symbol_dim_union_set_[lhs_root] = rhs_root;
    }
    product_equality_map_updated_ = false;
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

  std::unordered_map<SymbolicDimOp, int, SymDimHasher> sym_proc_map;
  for (SymbolicDimOp sym : new_rhs.symbols) ++sym_proc_map[sym];

  for (SymbolicDimOp sym : new_lhs.symbols) {
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
  for (auto& pair_outter : product_equality_map_) {
    const SymbolicDimProduct& x = pair_outter.first;
    auto factor_x = SymbolicDimProductDivide(lhs, x);
    if (!factor_x) continue;
    for (auto& pair_inner : pair_outter.second) {
      if (!pair_inner.second) continue;
      const SymbolicDimProduct& y = pair_inner.first;
      auto factor_y = SymbolicDimProductDivide(rhs, y);
      if (!factor_y || (*factor_x) != (*factor_y)) continue;
      return true;
    }
  }

  return false;
}

bool SymbolicDimMgr::UpdateProductEqualityMap() {
  // Return true if nothing is updated.
  if (product_equality_map_updated_) return true;

  SymbolicDimProductMap new_map;
  std::unordered_set<SymbolicDimProduct, SymProductHasher> product_set;
  for (auto& pair_outter : product_equality_map_) {
    const SymbolicDimProduct& x = pair_outter.first;
    for (auto& pair_inner : pair_outter.second) {
      if (!pair_inner.second) continue;

      const SymbolicDimProduct& y = pair_inner.first;
      SymbolicDimProduct new_x, new_y;
      std::tie(new_x, new_y) = SimplifySymbolicDimProductPair(x, y);
      if (new_x == new_y) continue;

      new_map[new_x][new_y] = new_map[new_y][new_x] = true;
      product_set.insert(new_x);
      product_set.insert(new_y);
    }
  }
  // hash function of SymbolicDimProduct is expensive, thus we map it to integer
  // domain first.
  std::unordered_map<const SymbolicDimProduct*, size_t> symProd2Idx;
  std::vector<const SymbolicDimProduct*> idx2SymProd(product_set.size());
  std::vector<size_t> idx2root(product_set.size());
  for (auto& x : product_set) {
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
    auto& rowMap = new_map[xProd];
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

      new_map[xSymProd][ySymProd] = new_map[ySymProd][xSymProd] = true;
    }

  product_equality_map_ = std::move(new_map);

  for (auto& x : product_set)
    for (auto& y : product_set) {
      if (!product_equality_map_[x][y]) continue;
      product_equality_map_[x][y] = product_equality_map_[y][x] = false;
      if (!IsMultipleOfKnownSymbolicDimProductEqualPair(x, y)) {
        product_equality_map_[x][y] = product_equality_map_[y][x] = true;
      }
    }

  std::unordered_set<SymbolicDimProduct, SymProductHasher> toRemove;
  for (auto& x : product_set) {
    if (std::all_of(product_set.begin(),
                    product_set.end(),
                    [&](const SymbolicDimProduct& y) {
                      return !product_equality_map_[x][y];
                    })) {
      toRemove.insert(x);
    }
  }

  for (auto& x : toRemove) {
    product_equality_map_.erase(x);
  }

  product_equality_map_updated_ = true;
  return true;
}

bool SymbolicDimMgr::IsSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                               const SymbolicDimProduct& rhs) {
  SymbolicDimProduct new_lhs, new_rhs;
  std::tie(new_lhs, new_rhs) = SimplifySymbolicDimProductPair(lhs, rhs);

  // Return true for identity case.
  if (new_lhs == new_rhs) return true;
  IR_ENFORCE(UpdateProductEqualityMap(), "Update product equality map failed.");
  return IsMultipleOfKnownSymbolicDimProductEqualPair(new_lhs, new_rhs);
}

}  // namespace pir
