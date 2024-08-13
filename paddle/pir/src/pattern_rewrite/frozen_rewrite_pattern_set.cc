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

#include "paddle/pir/include/pattern_rewrite/frozen_rewrite_pattern_set.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>

#include "paddle/pir/include/core/op_info.h"

namespace pir {

FrozenRewritePatternSet::FrozenRewritePatternSet()
    : impl_(std::make_shared<Impl>()) {}

FrozenRewritePatternSet::FrozenRewritePatternSet(
    RewritePatternSet&& patterns,
    const std::vector<std::string>& disabled_pattern_labels,
    const std::vector<std::string>& enabled_pattern_labels)
    : impl_(std::make_shared<Impl>()) {
  std::set<std::string> disabled_patterns, enabled_patterns;
  disabled_patterns.insert(disabled_pattern_labels.begin(),
                           disabled_pattern_labels.end());
  enabled_patterns.insert(enabled_pattern_labels.begin(),
                          enabled_pattern_labels.end());

  pir::OpInfoMap op_info_map;
  auto AddToOpsWhen = [&](std::unique_ptr<RewritePattern>& pattern,
                          std::function<bool(OpInfo)> callback) {
    if (op_info_map.empty())
      op_info_map = pattern->ir_context()->registered_op_info_map();
    for (auto& info_map : op_info_map) {
      if (callback(info_map.second))
        impl_->op_specific_native_pattern_map_[info_map.second].push_back(
            pattern.get());
      impl_->op_specific_native_patterns_.push_back(std::move(pattern));
    }
  };

  for (std::unique_ptr<RewritePattern>& pat : patterns.native_patterns()) {
    // Don't add patterns that haven't been enabled by the user.
    if (!enabled_patterns.empty()) {
      auto IsEnableFn = [&](const std::string& label) {
        return enabled_patterns.count(label);
      };
      if (!IsEnableFn(pat->debug_name()) &&
          std::none_of(pat->debug_labels().begin(),
                       pat->debug_labels().end(),
                       IsEnableFn))
        continue;
    }

    // Don't add patterns that have been disabled by the user.
    if (!disabled_patterns.empty()) {
      auto IsDisabledFn = [&](const std::string& label) {
        return disabled_patterns.count(label);
      };
      if (IsDisabledFn(pat->debug_name()) ||
          std::any_of(pat->debug_labels().begin(),
                      pat->debug_labels().end(),
                      IsDisabledFn))
        continue;
    }

    if (std::optional<OpInfo> root_name = pat->root_kind()) {
      impl_->op_specific_native_pattern_map_[*root_name].push_back(pat.get());
      impl_->op_specific_native_patterns_.push_back(std::move(pat));
      continue;
    }

    if (std::optional<TypeId> interface_id = pat->GetRootInterfaceID()) {
      AddToOpsWhen(
          pat, [&](OpInfo info) { return info.HasInterface(*interface_id); });
      continue;
    }

    if (std::optional<TypeId> trait_id = pat->GetRootTraitID()) {
      AddToOpsWhen(pat, [&](OpInfo info) { return info.HasTrait(*trait_id); });
      continue;
    }

    impl_->match_any_op_native_patterns_.push_back(std::move(pat));
  }
}

}  // namespace pir
