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

bool InitVariables(const std::vector<std::string>& var_list) {
  // write streams of Variable to server
  ClientContext context;
  VoidMessage void_ret;
  std::unique_ptr<ClientWriter<VariableMessage>> writer(
      stub_->InitVariables(&context, &void_ret));
  // send vars in scope to server using this stream.
  for (auto n = var_list.begin(); n != var_list.end(); n++) {
    auto* var = scope_.FindVar(*n);
    // TODO(typhoonzero): serialize by type.
    auto* tensor = var->Get<framework::LoDTensor>();
    VariableMessage msg;
    msg.varname = *n;
    std::ostringstream oss;
    framework::SerializeToStream(oss, *tensor);
    // FIXME(typhoonzero): no copy
    msg.serialized = oss.str();
    writer->Write(msg);
  }
  return true;
}

bool SendVariable(const framework::Variable* var) {
  // ClientContext context;
  // stub_->SendVariable(&context, )
  return true;
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
