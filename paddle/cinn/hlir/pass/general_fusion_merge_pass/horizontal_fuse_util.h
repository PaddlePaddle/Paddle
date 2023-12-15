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
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass_utils.h"

namespace cinn {
namespace hlir {
namespace pass {

using OpGroupPtr = api::OpGroup;
using framework::OpPatternKind;

template <typename FusePassCtxT>
struct HorizontalFuseUtil {
  using KindKeyT = std::pair<OpPatternKind, OpPatternKind>;

  static bool DetectFusabilityByKind(FusePassCtxT* ctx,
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

  typedef bool (*ConditionT)(FusePassCtxT* ctx,
                             const OpGroupPtr& src,
                             const OpGroupPtr& dst);

  static const std::map<KindKeyT, ConditionT>& GetConditionMap() {
    thread_local static std::map<KindKeyT, ConditionT> map(RawConditionMap());
    return map;
  }

  static std::map<KindKeyT, ConditionT> RawConditionMap() {
    return std::map<KindKeyT, ConditionT>{
        {{OpPatternKind::kElementWise, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kElementWise, framework::kReduction},
         &HorizontalElementwiseFuseReduce},

        {{OpPatternKind::kBroadcast, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kBroadcast, framework::kReduction}, &IsSameSize},

        {{OpPatternKind::kInjective, framework::kElementWise}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kInjective, framework::kReduction}, &IsSameSize},

        {{OpPatternKind::kReduction, framework::kElementWise},
         &HorizontalElementwiseFuseReduce},
        {{OpPatternKind::kReduction, framework::kBroadcast}, &IsSameSize},
        {{OpPatternKind::kReduction, framework::kInjective}, &IsSameSize},
        {{OpPatternKind::kReduction, framework::kReduction}, &ReduceFuseReduce},
    };
  }

  static bool IsSameSize(FusePassCtxT* ctx,
                         const OpGroupPtr& src,
                         const OpGroupPtr& dst) {
    return utils::IsSameSize(src, dst);
  }

  static bool HorizontalElementwiseFuseReduce(FusePassCtxT* ctx,
                                              const OpGroupPtr& src,
                                              const OpGroupPtr& dst) {
    // if same shape with horizontal relation
    if (IsSameSize(ctx, src, dst)) {
      return true;
    }

    const OpGroupPtr* ele_group = nullptr;
    const OpGroupPtr* reduce_group = nullptr;

    if (src.kind() == framework::kReduction) {
      ele_group = &dst;
      reduce_group = &src;
    } else {
      ele_group = &src;
      reduce_group = &dst;
    }

    size_t size_ele =
        utils::GetMasterNode(*ele_group).outputs()[0].shape().numel();

    bool can_fuse = false;
    reduce_group->WalkOpNodes([&](const api::OpNode& op) {
      if (op.kind() == OpPatternKind::kReduction) {
        size_t size_master = op.outputs()[0].shape().numel();
        if (size_ele == size_master) {
          can_fuse = true;
        }
      }
    });

    return can_fuse;
  }

  static bool ReduceFuseReduce(FusePassCtxT* ctx,
                               const OpGroupPtr& src,
                               const OpGroupPtr& dst) {
    return ctx->fuse_helper().ReduceFuseReduce(src, dst);
  }
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
