/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "send_recv_impl.h"

namespace paddle {
namespace operators {
namespace detail {

Status SendRecvServerImpl::SendVariable(ServerContext *context,
                                        const VariableMessage *in_var,
                                        VoidMessage *out_var) {
  // TODO(typhoonzero): support different variable types.
  std::istringstream iss(in_var->serialized());
  framework::LoDTensor t;
  framework::DeserializeFromStream(iss, &t);
  TensorWithName tensor_with_name =
      std::make_pair(in_var->varname(), std::move(t));

  var_recv_queue_.Push(std::move(tensor_with_name));
  return Status::OK;
}

Status SendRecvServerImpl::GetVariable(ServerContext *context,
                                       const VariableMessage *in_var,
                                       VariableMessage *out_var) {
  std::string get_var_name = in_var->varname();
  auto *var = scope_->FindVar(get_var_name);
  auto tensor = var->Get<framework::LoDTensor>();
  std::ostringstream oss;
  framework::SerializeToStream(oss, tensor, platform::CPUDeviceContext());

  std::string *varname = out_var->mutable_varname();
  *varname = get_var_name;
  std::string *serialized = out_var->mutable_serialized();
  *serialized = oss.str();
  return Status::OK;
}

Status SendRecvServerImpl::Wait(ServerContext *context,
                                const VoidMessage *in_var,
                                VoidMessage *out_var) {
  {
    std::unique_lock<std::mutex> lock(this->mutex_);
    condition_.wait(lock, [=] { return this->done_ == true; });
  }
  return Status::OK;
}

void SendRecvServerImpl::Reset() {
  std::lock_guard<std::mutex> lock(this->mutex_);
  done_ = false;
}

void SendRecvServerImpl::Done() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    done_ = true;
  }
  condition_.notify_all();
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
