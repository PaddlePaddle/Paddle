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

#pragma once

#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_bytebuffer_stream.h"
#include "paddle/fluid/operators/distributed/variable_response.h"

namespace grpc {
class ByteBuffer;
}  // namespace grpc
namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

using VarMsg = sendrecv::VariableMessage;
using MultiVarMsg = sendrecv::MultiVariableMessage;

class GRPCVariableResponse : public VariableResponse {
 public:
  GRPCVariableResponse(const framework::Scope* scope,
                       const platform::DeviceContext* dev_ctx,
                       bool create_scope = false)
      : VariableResponse(scope, dev_ctx, create_scope) {}

  virtual ~GRPCVariableResponse() {}

  int Parse(Source* source) override;

  // return:
  // 0:ok.
  // -1: unkown error.
  // other: number of error field.
  int Parse(const ::grpc::ByteBuffer& byte_buffer);
};

class GRPCMultiVariableResponseHelper {
 public:
  GRPCMultiVariableResponseHelper(const framework::Scope* scope,
                                  const platform::DeviceContext* dev_ctx,
                                  bool create_scope = false) {
    if (create_scope) {
      local_scope_ = scope->NewTmpScope().release();
    }
  }

  ~GRPCMultiVariableResponseHelper() {
    if (local_scope_) {
      delete local_scope_;
      local_scope_ = nullptr;
    }
  }

  inline framework::Scope* GetMutableLocalScope() const { return local_scope_; }

 protected:
  const framework::Scope* scope_;
  const platform::DeviceContext* dev_ctx_;
  bool create_scope_ = false;
  framework::Scope* local_scope_ = nullptr;
};

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
