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

#include <string>
#include <thread>
#include <unordered_map>

#ifdef PADDLE_WITH_DISTRIBUTE
#include "brpc/channel.h"
#include "brpc/server.h"
#endif

#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace distributed {

class Carrier;

class MessageBus final {
 public:
  MessageBus() = delete;

  explicit MessageBus(
      const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
      const std::unordered_map<int64_t, std::string>& rank_to_addr,
      const std::string& addr)
      : interceptor_id_to_rank_(interceptor_id_to_rank),
        rank_to_addr_(rank_to_addr),
        addr_(addr) {}

  ~MessageBus();

  // called by Interceptor, send InterceptorMessage to dst
  bool Send(const InterceptorMessage& interceptor_message);

  DISABLE_COPY_AND_ASSIGN(MessageBus);

 private:
  // function keep listen the port and handle the message
  void ListenPort();

  // check whether the dst is the same rank or different rank with src
  bool DstIsSameRank(int64_t src_id, int64_t dst_id);

#ifdef PADDLE_WITH_DISTRIBUTE
  // send the message inter rank (dst is different rank with src)
  bool SendInterRank(const InterceptorMessage& interceptor_message);
#endif

  // send the message intra rank (dst is the same rank with src)
  bool SendIntraRank(const InterceptorMessage& interceptor_message);

  // handed by above layer, save the info mapping interceptor id to rank id
  std::unordered_map<int64_t, int64_t> interceptor_id_to_rank_;

  // handed by above layer, save the info mapping rank id to addr
  std::unordered_map<int64_t, std::string> rank_to_addr_;

  // the ip needs to be listened
  std::string addr_;

#ifdef PADDLE_WITH_DISTRIBUTE
  // brpc server
  brpc::Server server_;
#endif

  // thread keeps listening to the port to receive remote message
  // this thread runs ListenPort() function
  std::thread listen_port_thread_;
};

}  // namespace distributed
}  // namespace paddle
