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

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_PSLIB)
#include "brpc/channel.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/fleet_executor/message_service.h"
#endif

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

class Carrier;

// A singleton MessageBus
class MessageBus final {
 public:
  MessageBus() = default;
  ~MessageBus();

  void Init(int64_t rank,
            const std::unordered_map<int64_t, std::string>& rank_to_addr,
            const std::string& addr);

  bool IsInit() const;

  // called by Interceptor, send InterceptorMessage to dst
  bool Send(int64_t dst_rank, const InterceptorMessage& interceptor_message);

  void IncreaseBarrierCount();
  void Barrier();
  bool DispatchMsgToCarrier(const InterceptorMessage& interceptor_message);

 private:
  DISABLE_COPY_AND_ASSIGN(MessageBus);

  // function keep listen the port and handle the message
  void ListenPort();

  const std::string& GetAddr(int64_t rank) const;

#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_PSLIB)
  // send the message inter rank (dst is different rank with src)
  bool SendInterRank(int64_t dst_rank,
                     const InterceptorMessage& interceptor_message);
#endif

  bool is_init_{false};

  int64_t rank_;

  // handed by above layer, save the info mapping rank id to addr
  std::unordered_map<int64_t, std::string> rank_to_addr_;

  // the ip needs to be listened
  std::string addr_;

#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_PSLIB)
  MessageServiceImpl message_service_;
  // brpc server
  brpc::Server server_;
#endif

  // for barrier
  std::mutex mutex_;
  std::condition_variable cv_;
  int count_{0};
};

}  // namespace distributed
}  // namespace paddle
