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

Status SendRecvServerImpl::InitVariables(
    ServerContext *context, ServerReader<VariableMessage> *in_var_reader,
    VoidMessage *void_ret) {
  // set up all variables to run server side block
  VariableMessage in_buf;
  while (in_var_reader->Read(&in_buf)) {
    // create var if not exist
    // auto *var = scope_->Var(in_buf.varname());
    // auto *tensor = var->GetMutable<framework::LoDTensor>();
    // std::istringstream iss(in_buf.serialized());
    // framework::DeserializeFromStream(iss, tensor);
  }
  *void_ret = VoidMessage();
  return Status::OK;
}

Status SendRecvServerImpl::SendVariable(ServerContext *context,
                                        const VariableMessage *in_var,
                                        VariableMessage *out_var) {
  framework::LoDTensor t;
  // TODO(typhoonzero): desirealize in_tensor and run pserver network.
  std::istringstream iss(in_var->serialized());
  framework::DeserializeFromStream(iss, &t);
  lodtensor_queue_.Push(std::move(t));
  // Block util the sub graph is done.
  t = lodtensor_return_queue_.Pop();
  std::ostringstream oss;
  // FIXME(typhoonzero): get context from op.
  framework::SerializeToStream(oss, t, platform::CPUDeviceContext());
  std::string *varname = out_var->mutable_varname();
  *varname = in_var->varname();
  std::string *serialized = out_var->mutable_serialized();
  *serialized = oss.str();

  return Status::OK;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
