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

#include "paddle/fluid/framework/details/all_gather_op_handle.h"

namespace paddle {
namespace framework {
namespace details {

AllGatherOpHandle::AllGatherOpHandle(const Scope &local_scope,
                                     const platform::Place &place)
    : local_scope_(local_scope), place_(place) {}

void AllGatherOpHandle::RunImpl() {
  PADDLE_ENFORCE_EQ(this->inputs_.size(), 1);
  auto &var_name = static_cast<VarHandle *>(this->inputs_[0])->name_;

  // Wait input done, this Wait is asynchronous operation
  auto &p = static_cast<VarHandle *>(this->inputs_[0])->place_;
  this->inputs_[0]->generated_op_->Wait(dev_ctxes_[p]);
  auto var = local_scope_.FindVar(var_name);
  PADDLE_ENFORCE(var);

  ParameterCollection::Instance().Get(var_name)->Send<Variable *>(&var);
}

std::string AllGatherOpHandle::Name() const { return "all_gather"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
