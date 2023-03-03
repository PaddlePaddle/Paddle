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

#include "Rewrite/FrozenRewritePatternSet.h"
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/TypeID.h"

namespace infra {

FrozenRewritePatternSet::FrozenRewritePatternSet()
    : impl_(std::make_shared<Impl>()) {}

FrozenRewritePatternSet::FrozenRewritePatternSet(
    RewritePatternSet&& patterns,
    llvm::ArrayRef<std::string> disabled_pattern_labels,
    llvm::ArrayRef<std::string> enabled_pattern_labels)
    : impl_(std::make_shared<Impl>()) {
  std::set<std::string> disabled_patterns, enabled_patterns;
  disabled_patterns.insert(disabled_pattern_labels.begin(),
                           disabled_pattern_labels.end());
  enabled_patterns.insert(enabled_pattern_labels.begin(),
                          enabled_pattern_labels.end());

  std::vector<mlir::RegisteredOperationName> op_infos;
  auto AddToOpsWhen =
      [&](std::unique_ptr<RewritePattern>& pattern,
          std::function<bool(mlir::RegisteredOperationName)> callback) {
        if (op_infos.empty())
          op_infos = pattern->GetContext()->getRegisteredOperations();
        for (mlir::RegisteredOperationName info : op_infos) {
          if (callback(info))
            impl_->native_op_specific_pattern_map[info].push_back(
                pattern.get());
          impl_->native_op_specific_pattern_list.push_back(std::move(pattern));
        }
      };

  for (std::unique_ptr<RewritePattern>& pat : patterns.GetNativePatterns()) {
    // Don't add patterns that haven't been enabled by the user.
    if (!enabled_patterns.empty()) {
      auto IsEnableFn = [&](llvm::StringRef label) {
        return enabled_patterns.count(label.str());
      };
      if (!IsEnableFn(pat->GetDebugName().str()) &&
          llvm::none_of(pat->GetDebugLabels(), IsEnableFn))
        continue;
    }

    // Don't add patterns that have been disabled by the user.
    if (!disabled_patterns.empty()) {
      auto IsDisabledFn = [&](llvm::StringRef label) {
        return disabled_patterns.count(label.str());
      };
      if (IsDisabledFn(pat->GetDebugName().str()) ||
          llvm::any_of(pat->GetDebugLabels(), IsDisabledFn))
        continue;
    }

    if (std::optional<mlir::OperationName> root_name = pat->GetRootKind()) {
      impl_->native_op_specific_pattern_map[*root_name].push_back(pat.get());
      impl_->native_op_specific_pattern_list.push_back(std::move(pat));
      continue;
    }

    if (std::optional<mlir::TypeID> interface_id = pat->GetRootInterfaceID()) {
      AddToOpsWhen(pat, [&](mlir::RegisteredOperationName info) {
        return info.hasInterface(*interface_id);
      });
      continue;
    }

    if (std::optional<mlir::TypeID> trait_id = pat->GetRootTraitID()) {
      AddToOpsWhen(pat, [&](mlir::RegisteredOperationName info) {
        return info.hasTrait(*trait_id);
      });
      continue;
    }

    impl_->native_any_op_patterns.push_back(std::move(pat));
  }
}

}  // namespace infra
