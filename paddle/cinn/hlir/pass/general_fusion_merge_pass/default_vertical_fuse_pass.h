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
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
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
      if (!DetectFusabilityByKind(ctx, producer, consumer)) {
        continue;
      }
      if (ctx->fuse_helper().DetectCycleIfFuse(producer, consumer)) {
        VLOG(4) << "Can't fuse because detect cycle";
        continue;
      }
      ctx->MarkFusible(producer, consumer);
    }
  }

  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;
  bool DetectFusabilityByKind(LightwareFusePassCtx* ctx,
                              const OpGroupPtr& src,
                              const OpGroupPtr& dst) const {
    const KindKeyT kind_pair(src.kind(), dst.kind());
    const auto& map = GetConditionMap();
    const auto& iter = map.find(kind_pair);
    if (iter == map.end()) {
      return false;
    }
    return iter->second(ctx, src, dst);
  }

  typedef bool (*ConditionT)(LightwareFusePassCtx* ctx,
                             const OpGroupPtr& src,
                             const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{OpPatternKind::kElementWise, framework::kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kElementWise, framework::kBroadcast},
         &DefaultVerticalFusePass::ElementwiseFuseBroadcast},
        {{OpPatternKind::kElementWise, framework::kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kElementWise, framework::kReduction},
         &DefaultVerticalFusePass::ElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, framework::kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kBroadcast},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kBroadcast, framework::kReduction},
         &DefaultVerticalFusePass::BroadcastFuseReduce},

        {{OpPatternKind::kInjective, framework::kElementWise},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, framework::kBroadcast},
         &DefaultVerticalFusePass::IsSameSize},
        {{OpPatternKind::kInjective, framework::kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kInjective, framework::kReduction},
         &DefaultVerticalFusePass::InjectiveHorizontalWithReduce},

        {{OpPatternKind::kReduction, framework::kElementWise},
         &DefaultVerticalFusePass::ReduceFuseElementwise},
        {{OpPatternKind::kReduction, framework::kBroadcast},
         &DefaultVerticalFusePass::ReduceFuseBroadcast},
        {{OpPatternKind::kReduction, framework::kInjective},
         &DefaultVerticalFusePass::HorizontalWithInjective},
        {{OpPatternKind::kReduction, framework::kReduction},
         &DefaultVerticalFusePass::ReduceFuseReduce},
    };
  }

  static bool IsSameSize(LightwareFusePassCtx* ctx,
                         const OpGroupPtr& src,
                         const OpGroupPtr& dst) {
    return utils::IsSameSize(src, dst);
  }

  static bool ElementwiseFuseBroadcast(LightwareFusePassCtx* ctx,
                                       const OpGroupPtr& src,
                                       const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseBroadcast(src, dst);
  }

  static bool HorizontalWithInjective(LightwareFusePassCtx* ctx,
                                      const OpGroupPtr& src,
                                      const OpGroupPtr& dst) {
    return ctx->fuse_helper().HorizontalWithInjective(src, dst);
  }

  static bool ElementwiseFuseReduce(LightwareFusePassCtx* ctx,
                                    const OpGroupPtr& src,
                                    const OpGroupPtr& dst) {
    return ctx->fuse_helper().ElementwiseFuseReduce(src, dst);
  }

  static bool BroadcastFuseReduce(LightwareFusePassCtx* ctx,
                                  const OpGroupPtr& src,
                                  const OpGroupPtr& dst) {
    return ctx->fuse_helper().BroadcastFuseReduce(src, dst);
  }

  static bool InjectiveHorizontalWithReduce(LightwareFusePassCtx* ctx,
                                            const OpGroupPtr& src,
                                            const OpGroupPtr& dst) {
    return ctx->fuse_helper().InjectiveHorizontalWithReduce(src, dst);
  }

  static bool ReduceFuseElementwise(LightwareFusePassCtx* ctx,
                                    const OpGroupPtr& src,
                                    const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseElementwise(src, dst);
  }

  static bool ReduceFuseBroadcast(LightwareFusePassCtx* ctx,
                                  const OpGroupPtr& src,
                                  const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseBroadcast(src, dst);
  }

  static bool ReduceFuseReduce(LightwareFusePassCtx* ctx,
                               const OpGroupPtr& src,
                               const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseReduce(src, dst);
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_FUSION_PASS(DefaultVerticalFusePass,
                          cinn::hlir::pass::DefaultVerticalFusePass);
