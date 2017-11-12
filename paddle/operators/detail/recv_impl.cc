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
    ServerContext *context,
    ServerReader<VariableMessage> *in_var_reader) override {
  // set up all variables to run server side block
  PADDLE_ENFORCE(scope_);
  VariableMessage in_buf;
  while (in_var_reader->Read(&in_buf)) {
    // create var if not exist
    auto *var = scope_->Var(in_buf.varname);
    auto *tensor = var->GetMutable<framework::LoDTensor>();
    std::istringstream iss(in_buf.serialized);
    framework::DeserializeFromStream(iss, *tensor);
  }
  return Status::OK;
}

Status SendRecvServerImpl::SendTensor(ServerContext *context,
                                      const std::string *in_tensor,
                                      std::string *out_tensor) override {
  framework::LodTensor t;
  // TODO(typhoonzero): desirealize in_tensor and run pserver network.
  std::istringstream iss(*in_tensor);
  framework::Tensor t;
  framework::DesirializeFromStream(iss, &t);
  lodtensor_queue_.Push(std::move(t));
  // Block util the sub graph is done.
  auto t = lodtensor_return_queue_.Pop();
  std::ostringstream oss;
  framework::SerializeToStream(oss, &t);
  *out_tensor = oss.str();

  return Status::OK;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
