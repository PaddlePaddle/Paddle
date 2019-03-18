//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace details {

struct FuseVarsOpHandle : public OpHandleBase {
 public:
  FuseVarsOpHandle(ir::Node *node, Scope *local_scope,
                   const platform::Place &place,
                   const std::unordered_map<std::string, int64_t> &inputs_numel,
                   const proto::VarType::Type var_type)
      : OpHandleBase(node),
        local_scope_(local_scope),
        place_(place),
        inputs_numel_(inputs_numel),
        type_(var_type) {
    total_numel_ = 0;
    for (auto in_numel : inputs_numel) {
      PADDLE_ENFORCE_GT(in_numel.second, 0);
      total_numel_ += in_numel.second;
    }
  }

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override { return false; };

 protected:
  void RunImpl() override;

 private:
  Scope *local_scope_;
  const platform::Place place_;
  const std::unordered_map<std::string, int64_t> inputs_numel_;
  const proto::VarType::Type type_;
  int64_t total_numel_;
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
