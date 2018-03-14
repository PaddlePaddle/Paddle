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

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>
#include <grpc++/support/byte_buffer.h>

namespace paddle {
namespace operators {
namespace detail {

enum class GrpcMethod {
  kSendVariable,
  kGetVariable,
};

static const int kGrpcNumMethods =
    static_cast<int>(GrpcMethod::kGetVariable) + 1;

inline const char* GrpcMethodName(GrpcMethod id) {
  switch (id) {
    case GrpcMethod::kSendVariable:
      return "/sendrecv.SendRecvService/SendVariable";
    case GrpcMethod::kGetVariable:
      return "/sendrecv.SendRecvService/GetVariable";
  }

  // Shouldn't be reached.
  PADDLE_ENFORCE(false, "Invalid id: not found valid method name");
  return nullptr;
}

class GrpcService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService() {
      for (int i = 0; i < kGrpcNumMethods; ++i) {
        AddMethod(new ::grpc::internal::RpcServiceMethod(
            GrpcMethodName(static_cast<GrpcMethod>(i)),
            ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
        ::grpc::Service::MarkMethodAsync(i);
      }
    }
    virtual ~AsyncService() {}

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace detail
}  // namespace operator
}  // namespace paddle
