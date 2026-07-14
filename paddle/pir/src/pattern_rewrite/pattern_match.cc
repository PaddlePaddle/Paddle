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

#include "paddle/pir/include/pattern_rewrite/pattern_match.h"

#include <algorithm>
#include <cstdint>

#include "paddle/common/enforce.h"
#include "paddle/pir/include/core/operation.h"

namespace pir {

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//
Pattern::Pattern(const std::string& root_name,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(context->GetRegisteredOpInfo(root_name),
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
    : Pattern(interface_id,
              RootKind::InterfaceId,
              generated_names,
              benefit,
              context) {}

Pattern::Pattern(MatchTraitOpTypeTag tag,
                 TypeId trait_id,
                 PatternBenefit benefit,
                 IrContext* context,
                 const std::vector<std::string>& generated_names)
    : Pattern(trait_id, RootKind::TraitId, generated_names, benefit, context) {}

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

void RewriterBase::ReplaceOp(Operation* op,
                             const std::vector<Value>& new_values) {
  // Notify that the rewriter subclass we're about to replace this root.
  NotifyRootReplaced(op, new_values);

  PADDLE_ENFORCE_EQ(
      op->num_results(),
      new_values.size(),
      common::errors::InvalidArgument("incorrect # of replacement values"));
  op->ReplaceAllUsesWith(new_values);
  for (uint32_t i = 0; i < op->num_results(); ++i) {
    NotifyValueReplaced(op->result(i), new_values[i]);
  }

  NotifyOperationRemoved(op);
  op->Erase();
}

void RewriterBase::EraseOp(Operation* op) {
  PADDLE_ENFORCE_EQ(op->use_empty(),
                    true,
                    common::errors::InvalidArgument(
                        "Erase op failed. op(%s)[id:%d] is used, the "
                        "expectation is that it is not used",
                        op->name(),
                        op->id()));
  NotifyOperationRemoved(op);
  op->Erase();
}

// Find uses of `from` and replace it with `to`.
void RewriterBase::ReplaceAllUsesWith(Value from, Value to) {
  for (auto it = from.use_begin(); it != from.use_end();) {
    UpdateRootInplace(it.owner(), [&]() { (it++)->set_source(to); });
  }
  NotifyValueReplaced(from, to);
}

}  // namespace pir
