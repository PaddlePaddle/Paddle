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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_bytebuffer_stream.h"
#include "paddle/fluid/operators/distributed/variable_response.h"

namespace paddle {
namespace operators {
namespace distributed {

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

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
