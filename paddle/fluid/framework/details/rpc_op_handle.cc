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

#include "paddle/fluid/framework/details/rpc_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

namespace paddle {
namespace framework {
namespace details {

RPCOpHandle::RPCOpHandle(ir::Node *node, const framework::OpDesc &op_desc,
                         Scope *local_scope, const std::string &name,
                         const platform::Place &place)
    : OpHandleBase(node),
      op_(framework::OpRegistry::CreateOp(op_desc)),
      local_scope_(local_scope),
      name_(name),
      place_(place) {}

void RPCOpHandle::RunImpl() {
  platform::RecordEvent record_event(
      Name(), platform::TracerEventType::Communication, 1);

  for (auto *in : inputs_) {
    auto &p = static_cast<VarHandle *>(in)->place();
    if (ir::IsControlDepVar(*in->Node())) {
      continue;
    }
    if (in->GeneratedOp()) {
      in->GeneratedOp()->RecordWaitEventOnCtx(dev_ctxes_.at(p));
    }
  }
  this->RunAndRecordEvent([this] { op_->Run(*local_exec_scopes_[0], place_); });
}

std::string RPCOpHandle::Name() const { return name_; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
