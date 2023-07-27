// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#pragma once

#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/fusion_pass_registrar.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/horizontal_fuse_util.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/input_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/input_fuse_pass_ctx.h"

namespace cinn {
namespace hlir {
namespace pass {

class DefaultInputFusePass final : public InputFusePass {
 public:
  DefaultInputFusePass() : InputFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(InputFusePassCtx* ctx) const override {
    const auto& consumer_set = ctx->PickConsumersWithSameInputs();

    const std::unordered_set<OpGroupPtr> consumer_candidates =
        [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& consumer : consumer_set) {
        if (consumer.kind() == framework::kElementWise ||
            consumer.kind() == framework::kBroadcast ||
            consumer.kind() == framework::kInjective ||
            consumer.kind() == framework::kReduction) {
          consumers.insert(consumer);
        }
      }
      return consumers;
    }();
    if (consumer_candidates.size() <= 1) {
      return;
    }

    std::vector<OpGroupList> fusionable_consumers;
    for (auto& candidate : consumer_candidates) {
      if (ctx->fuse_helper().IsConsumerSetsReachable(candidate,
                                                     consumer_candidates)) {
        continue;
      }
      if (fusionable_consumers.empty()) {
        fusionable_consumers.push_back({candidate});
        continue;
      }
      // check each fusionable groups
      bool fusionable = false;
      for (auto& groups : fusionable_consumers) {
        auto& last = groups.back();
        if (!HorizontalFuseUtil<InputFusePassCtx>::DetectFusabilityByKind(
                ctx, candidate, last)) {
          continue;
        }
        groups.push_back(candidate);
        fusionable = true;
        break;
      }

      // if can't fuse to othors Groups, new Groups.
      if (!fusionable) {
        fusionable_consumers.push_back({candidate});
      }
    }

    for (const auto& groups : fusionable_consumers) {
      if (groups.size() > 1) {
        ctx->MarkFusible(groups);
      }
    }
    VLOG(1) << "DefaultInputFusePass Finish";
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_FUSION_PASS(DefaultInputFusePass,
                          cinn::hlir::pass::DefaultInputFusePass);
