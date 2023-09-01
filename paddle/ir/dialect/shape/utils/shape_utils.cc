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

}  // namespace ir
