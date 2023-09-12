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
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass_ctx.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/vertical_fuse_pass.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/vertical_fuse_util.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass_utils.h"

namespace cinn {
namespace hlir {
namespace pass {

class DefaultVerticalFusePass final : public VerticalFusePass {
 public:
  DefaultVerticalFusePass() : VerticalFusePass() {}

  int Benefit() const override { return 100; }

  void operator()(LightwareFusePassCtx* ctx) const override {
    const auto& producer = ctx->PickOpGroup();
    const OpGroupList consumers = [&]() {
      OpGroupList consumers;
      for (const auto& consumer : producer.consumers()) {
        consumers.push_back(consumer);
      }
      return consumers;
    }();
    if (consumers.size() == 0) {
      return;
    }

    std::vector<OpGroupPtr> candidates;
    for (int i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!VerticalFuseUtil::DetectFusabilityByKind(ctx, producer, consumer)) {
        break;
      }
      candidates.push_back(consumer);
    }
    if (candidates.size() == consumers.size() &&
        producer.kind() == framework::kElementWise) {
      return;
    }

    for (int i = 0; i < consumers.size(); ++i) {
      const auto& consumer = consumers.at(i);
      if (!VerticalFuseUtil::DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        VLOG(4) << "Can't fuse because detect cycle";
        continue;
      }
      ctx->MarkFusible(producer, consumer);
    }
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_FUSION_PASS(DefaultVerticalFusePass,
                          cinn::hlir::pass::DefaultVerticalFusePass);
