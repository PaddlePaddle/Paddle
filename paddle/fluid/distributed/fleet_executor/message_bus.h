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

#pragma once

#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
#include "brpc/channel.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message_service.h"
#endif

#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

class Carrier;

// A singleton MessageBus
class MessageBus final {
 public:
  static MessageBus& Instance() {
    static MessageBus msg_bus;
    return msg_bus;
  }

  void Init(const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
            const std::unordered_map<int64_t, std::string>& rank_to_addr,
            const std::string& addr);

  bool IsInit() const;

  // called by Interceptor, send InterceptorMessage to dst
  bool Send(const InterceptorMessage& interceptor_message);

  ~MessageBus();

  DISABLE_COPY_AND_ASSIGN(MessageBus);

 private:
  MessageBus() = default;

  // function keep listen the port and handle the message
  void ListenPort();

  // check whether the dst is the same rank or different rank with src
  bool IsSameRank(int64_t src_id, int64_t dst_id);

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  // send the message inter rank (dst is different rank with src)
  bool SendInterRank(const InterceptorMessage& interceptor_message);
#endif

  // send the message intra rank (dst is the same rank with src)
  bool SendIntraRank(const InterceptorMessage& interceptor_message);

  bool is_init_{false};
  std::once_flag once_flag_;

  // handed by above layer, save the info mapping interceptor id to rank id
  std::unordered_map<int64_t, int64_t> interceptor_id_to_rank_;

  // handed by above layer, save the info mapping rank id to addr
  std::unordered_map<int64_t, std::string> rank_to_addr_;

  // the ip needs to be listened
  std::string addr_;

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  InterceptorMessageServiceImpl interceptor_message_service_;
  // brpc server
  brpc::Server server_;
#endif
};

}  // namespace distributed
}  // namespace paddle
