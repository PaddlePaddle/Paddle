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

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include <string>
#include "paddle/common/flags.h"

COMMON_DECLARE_bool(allreduce_record_one_event);

namespace paddle::framework::details {
struct VarHandleBase;

ComputationOpHandle::ComputationOpHandle(ir::Node *node,
                                         Scope *scope,
                                         phi::Place place,
                                         size_t scope_idx)
    : OpHandleBase(node),
      op_(framework::OpRegistry::CreateOp(*node->Op())),
      scope_(scope),
      place_(place),
      scope_idx_(scope_idx) {}

void ComputationOpHandle::RunImpl() {
  if (!FLAGS_allreduce_record_one_event) {
    WaitInputVarGenerated(place_);
  }

  auto run_func = [this]() { op_->Run(*local_exec_scopes_[0], place_); };

  if (is_lock_and_record_event_free_ || FLAGS_allreduce_record_one_event) {
    run_func();
  } else {
    this->RunAndRecordEvent(run_func);
  }
}

bool ComputationOpHandle::NeedWait(VarHandleBase *in_var) {
  bool need_wait =
      in_var && in_var->GeneratedOp() &&
      in_var->GeneratedOp()->DeviceContext(place_) != dev_ctxes_.at(place_);
  return need_wait;
}

std::string ComputationOpHandle::Name() const { return op_->Type(); }
}  // namespace paddle::framework::details
