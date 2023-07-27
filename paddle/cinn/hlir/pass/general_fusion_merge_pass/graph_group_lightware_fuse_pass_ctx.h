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

#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/graph_group_fuse_helper.h"
#include "paddle/cinn/hlir/pass/general_fusion_merge_pass/lightware_fuse_pass_ctx.h"

namespace cinn {
namespace hlir {
namespace pass {

class GraphGroupLightwareFusePassCtx;

class GraphGroupLightwareFusePassCtx final : public LightwareFusePassCtx {
 public:
  GraphGroupLightwareFusePassCtx(
      const FusionHelperBase* graph_group_fusion_helper,
      const OpGroupPtr& group,
      const std::function<void(const OpGroupPtr& first,
                               const OpGroupPtr& second)>& MarkFusible)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        group_(group),
        MarkFusible_(MarkFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  GraphGroupLightwareFusePassCtx(
      const FusionHelperBase* graph_group_fusion_helper,
      const OpGroupPtr& group,
      const std::function<void(const OpGroupList& candidates)>&
          MarkGroupListFusible)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        group_(group),
        MarkGroupListFusible_(MarkGroupListFusible),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  const OpGroupPtr& PickOpGroup() const override { return group_; }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void MarkFusible(const OpGroupPtr& first, const OpGroupPtr& second) override {
    MarkFusible_(first, second);
  }

  void MarkFusible(const OpGroupList& candidates) override {
    MarkGroupListFusible_(candidates);
  }

  const FusionHelperBase& graph_group_fusion_helper() const {
    return *graph_group_fusion_helper_;
  }

 private:
  const FusionHelperBase* graph_group_fusion_helper_;
  const OpGroupPtr& group_;
  const std::function<void(const OpGroupPtr& first, const OpGroupPtr& second)>
      MarkFusible_;
  const std::function<void(const OpGroupList& candidates)>
      MarkGroupListFusible_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

}  // namespace pass
}  // namespace hlir
}  // namespace cinn
