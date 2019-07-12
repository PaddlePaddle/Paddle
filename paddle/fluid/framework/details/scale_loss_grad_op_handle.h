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

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {
namespace details {

struct ScaleLossGradOpHandle : public OpHandleBase {
  ScaleLossGradOpHandle(ir::Node *node, size_t num_dev, Scope *scope,
                        platform::Place place, platform::DeviceContext *context,
                        proto::VarType::Type dtype);

  ~ScaleLossGradOpHandle() final;

  std::string Name() const override;

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return {scope_}; }

 private:
  float coeff_;
  Scope *scope_;
  platform::Place place_;
  proto::VarType::Type out_dtype_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
