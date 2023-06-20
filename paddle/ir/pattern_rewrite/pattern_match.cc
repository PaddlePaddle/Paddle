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

#include "paddle/ir/pattern_rewrite/pattern_match.h"

#include <algorithm>
#include <cassert>
#include <cstdint>

#include "paddle/ir/core/operation.h"

namespace ir {

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//
Pattern::Pattern(const std::string& root_name,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(context->GetRegisteredOpInfo(root_name).AsOpaquePointer(),
              RootKind::OperationInfo,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(MatchAnyOpTypeTag tag,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(nullptr, RootKind::Any, generated_names, benefit, context) {}

Pattern::Pattern(MatchInterfaceOpTypeTag tag,
                 TypeId interface_id,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(interface_id.AsOpaquePointer(),
              RootKind::InterfaceId,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(MatchTraitOpTypeTag tag,
                 TypeId trait_id,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(trait_id.AsOpaquePointer(),
              RootKind::TraitId,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(void* root_val,
                 RootKind root_kind,
                 const std::vector<std::string>& generated_names,
                 PatternBenefit benefit,
                 IrContext* context)
    : root_val_(root_val),
      root_kind_(root_kind),
      benefit_(benefit),
      context_(context) {
  if (generated_names.empty()) return;

  generated_ops_.reserve(generated_names.size());
  std::transform(generated_names.begin(),
                 generated_names.end(),
                 std::back_inserter(generated_ops_),
                 [context](const std::string& name) {
                   return context->GetRegisteredOpInfo(name);
                 });
}

RewritePattern::~RewritePattern() = default;

//===----------------------------------------------------------------------===//
// RewriterBase
//===----------------------------------------------------------------------===//
RewriterBase::~RewriterBase() = default;

// TODO(wilber): value support replace method.
// void RewriterBase::ReplaceOpWithIf(Operation* op,
//                                    ValueRange new_values,
//                                    bool* all_uses_replaced,
//                                    std::function<bool(OpOperand&)> functor) {
//   // assert(op->num_results() == new_values.size() && "incorrect number of
//   values to replace operation"); NotifyRootReplaced(op, new_values); bool
//   replace_all_uses = true; for (uint32_t i = 0; i < op->num_results(); ++i) {
//     // op->result(0)
//   }
// }
// void RewriterBase::ReplaceOpWithIf(Operation* op,
//                        ValueRange new_values,
//                        std::function<bool(OpOperand&)> functor) {
//   ReplaceOpWithIf(op, new_values, nullptr, functor);
// }

// TODO(wilber): support erase.
// void ReplaceOp(Operation* op, ValueRange new_values) {
//   NotifyRootReplaced(op, new_values);
//   assert(op->num_results() == new_values.size() && "incorrect # of
//   replacement values"); op->ReplaceAllUsesWith(new_values);
//   NotifyOperationRemoved(op);
//   op->erase();
// }
void RewriterBase::EraseOp(Operation* op) {
  //   assert(op->use_empty() && "expected 'op' to have no uses");
  //   NotifyOperationRemoved(op);
  //   op->erase();
}

void RewriterBase::ReplaceAllUsesWith(Value from, Value to) {
  // from.
  // for (OpOperand& operand : llvm::make_early_inc_range(from.getUses()))
  // {
  //   Operation* op = operand.getOwner();
  //   UpdateRootInPlace(op, [&]() { operand.set(to); });
  // }
}

// TODO(wilber): iterator maybe should support modify inplace.
void RewriterBase::ReplaceUseIf(Value from,
                                Value to,
                                std::function<bool(OpOperand&)> functor) {
  // for (auto it = from.begin(); it != from.end(); ++it) {
  //   // TODO: need a lvalue.
  //   if (functor(it.get())) {
  //     UpdateRootInplace(it.owner(), [&](){it.get().set(to)});
  //   }
  // }
}

void RewriterBase::ReplaceOpWithResultsOfAnotherOp(Operation* op,
                                                   Operation* new_op) {
  assert(op->num_results() == new_op->num_results() &&
         "replacement op doesn't match results of original op");
  // TODO(wilber): Op support results method.
  // if (op->num_results() == 1) return ReplaceOp(op,
  // new_op->result(0)); return ReplaceOp(op, new_op->GetResults());
}

}  // namespace ir
