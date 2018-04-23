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

#include "paddle/fluid/framework/details/send_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

SendOpHandle::SendOpHandle(const framework::OpDesc &op_desc,
                           const Scope *local_scope,
                           const platform::Place &place)
    : op_(framework::OpRegistry::CreateOp(op_desc)),
      local_scope_(local_scope),
      place_(place) {}

void SendOpHandle::RunImpl() {
  // Wait input done
  for (auto *in : inputs_) {
    auto &p = static_cast<VarHandle *>(in)->place_;
    if (in->DebugString() == "dummy") {  // HACK
      continue;
    }
    in->generated_op_->Wait(dev_ctxes_[p]);
  }
  auto &tmp_scope = local_scope_->FindVar(kLocalExecScopeName)->Get<Scope *>();
  // FIXME(wuyi): can not use RunAndRecordEvent here, for it will cause dead
  // lock.
  op_->Run(*tmp_scope, place_);
}

std::string SendOpHandle::Name() const { return "send"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
