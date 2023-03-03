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

#include "IR/PatternMatch.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

namespace infra {

//===----===//
// PatternBenefit
//===----===//

PatternBenefit::PatternBenefit(unsigned benefit) : representation(benefit) {
  assert(representation == benefit && benefit != ImpossibleToMatchSentinel &&
         "This pattern match benefit is too large to represent");
}

//===----===//
// Pattern
//===----===//

Pattern::Pattern(llvm::StringRef root_name,
                 PatternBenefit benefit,
                 mlir::MLIRContext* context,
                 llvm::ArrayRef<llvm::StringRef> generated_names)
    : Pattern(mlir::OperationName(root_name, context).getAsOpaquePointer(),
              RootKind::OperationName,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(MatchAnyOpTypeTag tag,
                 PatternBenefit benefit,
                 mlir::MLIRContext* context,
                 llvm::ArrayRef<llvm::StringRef> generated_names)
    : Pattern(nullptr, RootKind::Any, generated_names, benefit, context) {}

Pattern::Pattern(MatchInterfaceOpTypeTag tag,
                 mlir::TypeID interface_id,
                 PatternBenefit benefit,
                 mlir::MLIRContext* context,
                 llvm::ArrayRef<llvm::StringRef> generated_names)
    : Pattern(interface_id.getAsOpaquePointer(),
              RootKind::InterfaceID,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(MatchTraitOpTypeTag tag,
                 mlir::TypeID trait_id,
                 PatternBenefit benefit,
                 mlir::MLIRContext* context,
                 llvm::ArrayRef<llvm::StringRef> generated_names)
    : Pattern(trait_id.getAsOpaquePointer(),
              RootKind::TraitID,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(const void* root_val,
                 RootKind root_kind,
                 llvm::ArrayRef<llvm::StringRef> generated_names,
                 PatternBenefit benefit,
                 mlir::MLIRContext* context)
    : root_val_(root_val),
      root_kind_(root_kind),
      benefit(benefit),
      context_(context),
      has_bound_recursion_(false) {
  if (generated_names.empty()) return;

  generated_ops_.reserve(generated_names.size());
  std::transform(generated_names.begin(),
                 generated_names.end(),
                 std::back_inserter(generated_ops_),
                 [context](llvm::StringRef name) {
                   return mlir::OperationName(name, context);
                 });
}

//===----===//
// RewriterBase
//===----===//

void RewriterBase::ReplaceOpWithIf(
    mlir::Operation* op,
    mlir::ValueRange new_values,
    bool* all_uses_replaced,
    std::function<bool(mlir::OpOperand&)> functor) {
  assert(op->getNumResults() == new_values.size() &&
         "incorrect number of values to replace operation");

  notifyRootReplaced(op, new_values);

  bool replace_all_uses = true;
  for (auto it : llvm::zip(op->getResults(), new_values)) {
    std::get<0>(it).replaceUsesWithIf(std::get<1>(it), functor);
    replace_all_uses &= std::get<0>(it).use_empty();
  }
  if (all_uses_replaced) *all_uses_replaced = replace_all_uses;
}

void RewriterBase::ReplaceOpWithIf(
    mlir::Operation* op,
    mlir::ValueRange new_values,
    std::function<bool(mlir::OpOperand&)> functor) {
  ReplaceOpWithIf(op, new_values, nullptr, functor);
}

void RewriterBase::ReplaceOp(mlir::Operation* op, mlir::ValueRange new_values) {
  notifyRootReplaced(op, new_values);

  assert(op->getNumResults() == new_values.size() &&
         "incorrect # of replacement values");
  op->replaceAllUsesWith(new_values);

  notifyOperationRemoved(op);
  op->erase();
}

void RewriterBase::EraseOp(mlir::Operation* op) {
  assert(op->use_empty() && "expected 'op' to have no uses");
  notifyOperationRemoved(op);
  op->erase();
}

void RewriterBase::ReplaceAllUsesWith(mlir::Value from, mlir::Value to) {
  for (mlir::OpOperand& operand : llvm::make_early_inc_range(from.getUses())) {
    mlir::Operation* op = operand.getOwner();
    UpdateRootInPlace(op, [&]() { operand.set(to); });
  }
}

void RewriterBase::ReplaceUseIf(mlir::Value from,
                                mlir::Value to,
                                std::function<bool(mlir::OpOperand&)> functor) {
  for (mlir::OpOperand& operand : llvm::make_early_inc_range(from.getUses())) {
    if (functor(operand))
      UpdateRootInPlace(operand.getOwner(), [&]() { operand.set(to); });
  }
}

void RewriterBase::ReplaceOpWithResultsOfAnotherOp(mlir::Operation* op,
                                                   mlir::Operation* new_op) {
  assert(op->getNumResults() == new_op->getNumResults() &&
         "replacement op doesn't match results of original op");
  // TODO(wilber): ....
  if (op->getNumResults() == 1) return ReplaceOp(op, new_op->getResult(0));
  return ReplaceOp(op, new_op->getResults());
}

}  // namespace infra
