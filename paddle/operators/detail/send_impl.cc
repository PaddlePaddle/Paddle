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
                             const std::string& inname,
                             const std::string& outname) {
  ClientContext context;
  VariableMessage msg, out_msg;
  // FIXME(typhoonzero): pass device context to here.
  auto ctx = platform::CPUDeviceContext();
  auto* var = scope.FindVar(inname);
  PADDLE_ENFORCE(var);
  // TODO(typhoonzero): support SelectedRows
  PADDLE_ENFORCE(var->IsType<framework::LoDTensor>(),
                 "Only support LoDTensor, %s has wrong type", inname);
  const framework::LoDTensor& tensor = var->Get<framework::LoDTensor>();
  std::ostringstream oss;
  framework::SerializeToStream(oss, tensor, ctx);
  msg.set_varname(inname);
  msg.set_serialized(oss.str());
  Status status = stub_->SendVariable(&context, msg, &out_msg);
  if (!status.ok()) {
    return false;
  }
  std::istringstream iss(out_msg.serialized());
  framework::LoDTensor ret_tensor;
  framework::DeserializeFromStream(iss, &ret_tensor);
  auto* outvar = scope.FindVar(outname);
  framework::LoDTensor* out_tensor = outvar->GetMutable<framework::LoDTensor>();
  // FIXME(typhoonzero): do not copy.
  framework::CopyFrom(ret_tensor, ctx.GetPlace(), ctx, out_tensor);
  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
