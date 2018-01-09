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

bool RPCClient::SendVariable(const framework::Scope& scope,
                             const std::string& inname) {
  ClientContext context;
  VariableMessage msg;
  VoidMessage out_msg;
  // FIXME(typhoonzero): pass device context to here.
  auto ctx = platform::CPUDeviceContext();
  auto* var = scope.FindVar(inname);
  PADDLE_ENFORCE(var);
  SerializeToMessage(inname, var, ctx, &msg);

  Status status = stub_->SendVariable(&context, msg, &out_msg);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC error: " << status.error_message();
    return false;
  }
  return true;
}

bool RPCClient::GetVariable(const framework::Scope& scope,
                            const std::string& outname) {
  ClientContext context;
  VariableMessage call_msg, ret_msg;
  call_msg.set_varname(outname);
  auto ctx = platform::CPUDeviceContext();
  Status status = stub_->GetVariable(&context, call_msg, &ret_msg);
  auto* outvar = scope.FindVar(outname);
  if (!status.ok()) {
    LOG(ERROR) << "gRPC error: " << status.error_message();
    return false;
  }

  std::istringstream iss(ret_msg.serialized());
  DeserializeFromMessage(ret_msg, ctx, outvar);

  return true;
}

void RPCClient::Wait() {
  ClientContext context;
  VoidMessage call_msg, ret_msg;
  stub_->Wait(&context, call_msg, &ret_msg);
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
