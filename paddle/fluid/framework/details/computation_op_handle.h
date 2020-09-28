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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class OperatorBase;
class Scope;
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {
struct VarHandleBase;

class ComputationOpHandle : public OpHandleBase {
 public:
  ComputationOpHandle(ir::Node *node, Scope *scope, platform::Place place,
                      size_t scope_idx);

  OperatorBase *GetOp() { return op_.get(); }

  const OperatorBase *GetOp() const { return op_.get(); }

  std::string Name() const override;

  const Scope *GetScope() const { return scope_; }

  Scope *GetScope() { return scope_; }

  const platform::Place &GetPlace() const { return place_; }

  void SetLockAndRecordEventFree(bool b) { is_lock_and_record_event_free_ = b; }

  size_t GetScopeIdx() const { return scope_idx_; }

 protected:
  void RunImpl() override;

  bool NeedWait(VarHandleBase *in_var) override;

  std::vector<Scope *> GetLocalScopes() override { return {scope_}; }

 private:
  std::unique_ptr<OperatorBase> op_;
  Scope *scope_;
  platform::Place place_;
  size_t scope_idx_;
  bool is_lock_and_record_event_free_{false};
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
