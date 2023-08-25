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
    if (!(op->dyn_cast<SymbolicDim>())) continue;
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
                               SymbolicDimProduct& product) -> LogicalResult {
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

bool SymbolicDimMgr::mapSymbolicDimProductEqual(const SymbolicDimProduct& lhs,
                                                const SymbolicDimProduct& rhs) {
  LLVM_DEBUG(llvm::dbgs() << "Try to map product equal: x = " << lhs
                          << "\ny = " << rhs << "\n");
  SymbolicDimProduct newLhs, newRhs;
  // std::tie(newLhs, newRhs) = simplifySymbolicDimProductPair(lhs, rhs);
  LLVM_DEBUG(llvm::dbgs() << "Try to map product equal after simplify: x = "
                          << newLhs << "\ny = " << newRhs << "\n");

  // early return for identity case.
  if (newLhs == newRhs) return success();

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
  return success();
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

// // ShapeAnalysis hold at ShapedType
// bool ShapeAnalysis::isSameNumElements(ir::Value lhs, ir::Value rhs) {
//   if (lhs == rhs) return true;

//   auto lhsTy = lhs.getType().dyn_cast<ShapedType>();
//   auto rhsTy = rhs.getType().dyn_cast<ShapedType>();

//   if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

//   return isProductEqual(lhs, 0, lhsTy.getRank(), rhs, 0, rhsTy.getRank());
// }

// bool ShapeAnalysis::isProductEqual(
//     Value lhs, int lhsFrom, int lhsTo, Value rhs, int rhsFrom, int rhsTo) {
//   std::vector<int> lhsDimIdxs, rhsDimIdxs;
//   lhsDimIdxs.reserve(lhsTo - lhsFrom);
//   rhsDimIdxs.reserve(rhsTo - rhsFrom);
//   for (int i = lhsFrom; i < lhsTo; ++i) lhsDimIdxs.push_back(i);
//   for (int i = rhsFrom; i < rhsTo; ++i) rhsDimIdxs.push_back(i);

//   return isProductEqual(lhs, lhsDimIdxs, rhs, rhsDimIdxs);
// }

// SymbolicDimShapeAnalysis::SymbolicDimShapeAnalysis(ir::Operation* op)
//     : op_(op), mgr_(op->GetParentOp()->dyn_cast<ir::ModuleOp>()) {
//   mgr_.load();
//   ir::ModuleOp m = op->GetParentOp()->dyn_cast<ir::ModuleOp>();
//   for (uint32_t r_idx = 0; idx < op->num_regions(); r_idx++) {
//     for (auto b_it = op->region(r_idx).begin(); b_it !=
//     op->region(r_idx).end();
//          b_it++) {
//       for (auto op_it = b_it->begin(); op_it != b_it->end(); op_it++) {
//         Value result = op->getResult(0);
//         auto resultTy =
//             result.getType().dyn_cast<paddle::dialect::DenseTensorType>();
//         if (!resultTy) return;

//         auto& symbols = value2SymDims_[result];
//         /*******************************************************/
//         /*   version 1: add SymbolDimAttr to DenseTensorType.  */
//         /*******************************************************/
//         // auto attrs = resultTy.getAttr<SymbolDimArrayAttr>();
//         // for (const auto& attr : attrs) {
//         //   auto symOp = mgr_.symbolTable().lookup<SymbolicDim>(
//         //       attr.cast<SymbolDimAttr>().getValue());
//         //   if (!symOp) continue;
//         //   symbols.push_back(symOp);
//         // }

//         /********************************************************************/
//         /*   version 2: map DenseTensor and SymbolicDim in a var globally. */
//         /********************************************************************/
//         unordered_map<ir::value, std::string> value2SymDimsName;  // Globally
//         if (value2SymDims_global.count(result)) {
//           for (auto str : value2SymDims_global[result]) {
//             auto symOp = mgr_.symbolTable().lookup<SymbolicDim>(str);
//             if (!symOp) continue;
//             symbols.push_back(symOp);
//           }
//         }
//       }
//     }
//   }
// }

// SymbolicDimShapeAnalysis::~SymbolicDimShapeAnalysis() { mgr_.save(); }

// bool SymbolicDimShapeAnalysis::isShapeEqual(ir::Value lhs, ir::Value rhs) {
//   if (lhs == rhs) return true;

//   auto lhsTy = lhs.getType().dyn_cast<ShapedType>();
//   auto rhsTy = rhs.getType().dyn_cast<ShapedType>();

//   if (!lhsTy || !rhsTy || !lhsTy.hasRank() || !rhsTy.hasRank()) return false;

//   if (lhsTy.hasStaticShape() && rhsTy.hasStaticShape()) {
//     return lhsTy.getShape() == rhsTy.getShape();
//   }

//   auto lhsIt = value2SymDims_.find(lhs);
//   auto rhsIt = value2SymDims_.find(rhs);

//   if (lhsIt == value2SymDims_.end() || rhsIt == value2SymDims_.end() ||
//       lhsIt->second.size() != rhsIt->second.size())
//     return false;

//   std::vector<SymbolicDim> lhsSyms;
//   std::vector<SymbolicDim> rhsSyms;
//   for (auto sym : lhsIt->second) {
//     lhsSyms.push_back(mgr_.getRootSymbolicDim(sym));
//   }
//   for (auto sym : rhsIt->second) {
//     rhsSyms.push_back(mgr_.getRootSymbolicDim(sym));
//   }
//   return lhsSyms == rhsSyms;
// }

// bool SymbolicDimShapeAnalysis::isProductEqual(Value lhs,
//                                               std::vector<int> lhsDimIdxs,
//                                               Value rhs,
//                                               std::vector<int> rhsDimIdxs) {
//   SymbolicDimProduct lhsProd;
//   SymbolicDimProduct rhsProd;

//   auto buildSymbolicDimProduct =
//       [&](SymbolicDimProduct& prod, Value value, std::vector<int> dimIdxs) {
//         auto ty = value.getType().dyn_cast<ShapedType>();
//         auto it = value2SymDims_.find(value);
//         if (!ty || !ty.hasRank()) return false;

//         for (int idx : dimIdxs) {
//           if (ty.getShape()[idx] == ShapedType::kDynamic) {
//             if (it == value2SymDims_.end() || it->second.size() <= idx)
//               return false;
//             prod.symbols.push_back(it->second[idx]);
//           } else {
//             prod.factor *= ty.getShape()[idx];
//           }
//         }
//         return true;
//       };

//   if (!buildSymbolicDimProduct(lhsProd, lhs, lhsDimIdxs) ||
//       !buildSymbolicDimProduct(rhsProd, rhs, rhsDimIdxs)) {
//     return false;
//   }

//   return mgr_.isSymbolicDimProductEqual(lhsProd, rhsProd);
// }

}  // namespace ir
