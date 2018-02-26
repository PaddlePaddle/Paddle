/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

namespace paddle {
namespace operators {
namespace detail {

void SerializeToMessage(const std::string& name, const framework::Variable* var,
                        const platform::DeviceContext& ctx,
                        sendrecv::VariableMessage* msg) {
  msg->set_varname(name);
  std::ostringstream oss;
  switch (framework::ToVarType(var->Type())) {
    case framework::proto::VarType_Type_LOD_TENSOR:
      msg->set_type(sendrecv::VarType::LOD_TENSOR);
      framework::SerializeToStream(oss, var->Get<framework::LoDTensor>(), ctx);
      break;
    case framework::proto::VarType_Type_SELECTED_ROWS:
      msg->set_type(sendrecv::VarType::SELECTED_ROWS);
      framework::SerializeToStream(oss, var->Get<framework::SelectedRows>(),
                                   ctx);
      break;
    default: {
      PADDLE_THROW("Serialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
  msg->set_serialized(oss.str());
}

void DeserializeFromMessage(const sendrecv::VariableMessage& msg,
                            const platform::DeviceContext& ctx,
                            framework::Variable* var) {
  std::istringstream iss(msg.serialized());
  switch (msg.type()) {
    case sendrecv::VarType::LOD_TENSOR:
      DeserializeFromStream(iss, var->GetMutable<framework::LoDTensor>(), ctx);
      break;
    case sendrecv::VarType::SELECTED_ROWS: {
      DeserializeFromStream(iss, var->GetMutable<framework::SelectedRows>(),
                            ctx);
      break;
    }
    default: {
      PADDLE_THROW("Deserialize does not support type: %s",
                   typeid(var->Type()).name());
      break;
    }
  }
}

}  // namespace detail
}  // namespace operators
}  // namespace paddle
