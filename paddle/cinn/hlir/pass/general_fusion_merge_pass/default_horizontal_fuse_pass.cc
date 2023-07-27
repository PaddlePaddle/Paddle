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
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/horizontal_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/horizontal_fuse_util.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass_ctx.h"

namespace cinn {
namespace hlir {
namespace pass {

class DefaultHorizontalFusePass final : public HorizontalFusePass {
 public:
  DefaultHorizontalFusePass() : HorizontalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer = ctx->PickOpGroup();
    const std::unordered_set<OpGroupPtr> consumer_candidates =
        [&]() -> std::unordered_set<OpGroupPtr> {
      std::unordered_set<OpGroupPtr> consumers;
      for (const auto& consumer : producer.consumers()) {
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
        if (!HorizontalFuseUtil<LightwareFusePassCtx>::DetectFusabilityByKind(
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
        // Trick for BERT, maybe not required, wait for substitution from
        // unordered_set to set
        if (groups.size() == 2) {
          OpGroupList fuse_group;
          if (groups[1].group_id().substr(0, 4) == "cast" &&
              groups[0].group_id() == "reshape_split") {
            fuse_group.push_back(groups[1]);
            fuse_group.push_back(groups[0]);
            ctx->MarkFusible(fuse_group);
            continue;
          }
        }
        ctx->MarkFusible(groups);
      }
    }
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_FUSION_PASS(DefaultHorizontalFusePass,
                          cinn::hlir::pass::DefaultHorizontalFusePass);
