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

#include "paddle/cinn/api/op_group.h"
#include "paddle/cinn/hlir/framework/op.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass_ctx.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass_utils.h"

namespace cinn {
namespace hlir {
namespace pass {

using OpGroupPtr = api::OpGroup;
using framework::OpPatternKind;

struct VerticalFuseUtil {
  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;

  static bool DetectFusabilityByKind(LightwareFusePassCtx* ctx,
                                     const OpGroupPtr& src,
                                     const OpGroupPtr& dst) {
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
        {{OpPatternKind::kElementWise, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kBroadcast},
         &ElementwiseFuseBroadcast},
        {{OpPatternKind::kElementWise, framework::kInjective},
         &HorizontalWithInjective},
        {{OpPatternKind::kElementWise, framework::kReduction},
         &ElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kInjective},
         &HorizontalWithInjective},
        {{OpPatternKind::kBroadcast, framework::kReduction},
         &BroadcastFuseReduce},

        {{OpPatternKind::kInjective, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kInjective},
         &HorizontalWithInjective},
        {{OpPatternKind::kInjective, framework::kReduction},
         &InjectiveHorizontalWithReduce},

        {{OpPatternKind::kReduction, framework::kElementWise},
         &ReduceFuseElementwise},
        {{OpPatternKind::kReduction, framework::kBroadcast},
         &ReduceFuseBroadcast},
        {{OpPatternKind::kReduction, framework::kInjective},
         &HorizontalWithInjective},
        {{OpPatternKind::kReduction, framework::kReduction}, &ReduceFuseReduce},
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
