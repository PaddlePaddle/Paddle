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

#include <memory>

#include "paddle/cinn/api/op_group.h"

namespace cinn {
namespace api {

class FusePassContext {
 public:
  virtual ~FusePassCtx() {}

  virtual void EnableFuse(const OpGroup& first, const OpGroup& second) = 0;

 protected:
  FusePassCtx() = default;
};

class LightwareFusePassCtx : public FusePassContext {
 public:
  virtual ~LightwareFusePassCtx() {}

  virtual const OpGroup& PickOpGroup() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void EnableFuse(const OpGroup& first, const OpGroup& second) = 0;

 protected:
  LightwareFusePassCtx() = default;
};

class GraphGroupLightwareFusePassCtx final : public LightwareFusePassCtx {
 public:
  GraphGroupLightwareFusePassCtx(
      const FusionHelperBase* graph_group_fusion_helper,
      const OpGroup& group,
      const std::function<void(const OpGroup& first, const OpGroup& second)>&
          EnableFuse)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        group_(group),
        EnableFuse_(EnableFuse),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupLightwareFusePassCtx>(this)) {}

  const OpGroup& PickOpGroup() const override { return group_; }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void EnableFuse(const OpGroup& first, const OpGroup& second) override {
    EnableFuse_(first, second);
  }

  const FusionHelperBase& graph_group_fusion_helper() const {
    return *graph_group_fusion_helper_;
  }

 private:
  const FusionHelperBase* graph_group_fusion_helper_;
  const OpGroup& group_;
  const std::function<void(const OpGroup& first, const OpGroup& second)>
      EnableFuse_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

class InputFusePassCtx : public FusePassCtx {
 public:
  virtual ~InputFusePassCtx() {}

  virtual const OpGroupList& PickConsumersWithSameInputs() const = 0;

  virtual const FuseHelper& fuse_helper() const = 0;

  virtual void EnableFuse(const OpGroup& first, const OpGroup& second) = 0;

 protected:
  InputFusePassCtx() = default;
};

class GraphGroupInputFusePassCtx final : public InputFusePassCtx {
 public:
  GraphGroupInputFusePassCtx(
      const FusionHelperBase* graph_group_fusion_helper,
      const OpGroupList& groups,
      const std::function<void(const OpGroup& first, const OpGroup& second)>&
          EnableFuse)
      : graph_group_fusion_helper_(graph_group_fusion_helper),
        groups_(groups),
        EnableFuse_(EnableFuse),
        fuse_helper_(
            new GraphGroupFuseHelper<GraphGroupInputFusePassCtx>(this)) {}

  const OpGroupList& PickConsumersWithSameInputs() const override {
    return groups_;
  }

  const FuseHelper& fuse_helper() const override { return *fuse_helper_; }

  void EnableFuse(const OpGroup& first, const OpGroup& second) override {
    EnableFuse_(first, second);
  }

  const FusionHelperBase& graph_group_fusion_helper() const {
    return *graph_group_fusion_helper_;
  }

 private:
  const FusionHelperBase* graph_group_fusion_helper_;
  const OpGroupList& groups_;
  const std::function<void(const OpGroup& first, const OpGroup& second)>
      EnableFuse_;
  const std::unique_ptr<const FuseHelper> fuse_helper_;
};

}  // namespace api
}  // namespace cinn
