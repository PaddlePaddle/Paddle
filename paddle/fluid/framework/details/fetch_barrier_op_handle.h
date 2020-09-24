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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class Scope;
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

// **NOTE**: fetch_barrier op is special it outputs all recved variables on
// all places if there are multiple places, must init with
// multiple dev_ctxes_ !!!!

struct VarHandleBase;

struct FetchBarrierOpHandle : public OpHandleBase {
 public:
  FetchBarrierOpHandle(ir::Node *node, const std::vector<Scope *> &local_scopes,
                       const std::vector<platform::Place> &places);

  bool IsMultiDeviceTransfer() override;

  std::string Name() const override;

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return local_scopes_; }

  bool NeedWait(VarHandleBase *in_var) override;

 private:
  std::unique_ptr<OperatorBase> op_;
  std::vector<Scope *> local_scopes_;
  std::vector<platform::Place> places_;
  Scope *run_scope_;
  platform::Place place_;

  bool is_lock_and_record_event_free_{false};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
