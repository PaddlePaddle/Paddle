// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
#pragma once

#include "brpc/server.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"

namespace paddle {
namespace distributed {

class MessageServiceImpl : public MessageService {
 public:
  MessageServiceImpl() {}
  virtual ~MessageServiceImpl() {}
  virtual void ReceiveInterceptorMessage(
      google::protobuf::RpcController* control_base,
      const InterceptorMessage* request, InterceptorResponse* response,
      google::protobuf::Closure* done);
  virtual void IncreaseBarrierCount(
      google::protobuf::RpcController* control_base,
      const InterceptorMessage* request, InterceptorResponse* response,
      google::protobuf::Closure* done);
};

}  // namespace distributed
}  // namespace paddle
#endif
