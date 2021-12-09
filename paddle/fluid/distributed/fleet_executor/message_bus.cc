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

#include <chrono>
#include <memory>
#include <set>
#include <thread>

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/fleet_executor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

namespace paddle {
namespace distributed {

void MessageBus::Init(
    const std::unordered_map<int64_t, int64_t>& interceptor_id_to_rank,
    const std::unordered_map<int64_t, std::string>& rank_to_addr,
    const std::string& addr) {
  PADDLE_ENFORCE_EQ(is_init_, false, platform::errors::AlreadyExists(
                                         "MessageBus is already init."));
  is_init_ = true;
  interceptor_id_to_rank_ = interceptor_id_to_rank;
  rank_to_addr_ = rank_to_addr;
  addr_ = addr;

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL)
  // NOTE: To make the brpc is compatible with collective,
  // need release the handler holding the ip address.
  if (addr_ != "") {
    VLOG(3) << "Message bus is releasing the fd held by gen_comm_id.";
    paddle::platform::SocketServer& socket_server =
        paddle::platform::SocketServer::GetInstance(addr_);
    int server_fd = socket_server.socket();
    if (server_fd != -1) {
      socket_server.Release();
    }
  }
#endif

  ListenPort();
}

bool MessageBus::IsInit() const { return is_init_; }

MessageBus::~MessageBus() {
  // NOTE: fleet_executor inits carrier before message bus,
  // therefore the message bus's destructor will be called first
  Carrier& carrier = Carrier::Instance();
  carrier.Release();
  VLOG(3) << "Message bus releases resource.";
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  server_.Stop(1000);
  server_.Join();
#endif
}

bool MessageBus::Send(const InterceptorMessage& interceptor_message) {
  // called by Interceptor, send InterceptorMessage to dst
  int64_t src_id = interceptor_message.src_id();
  int64_t dst_id = interceptor_message.dst_id();
  if (IsSameRank(src_id, dst_id)) {
    VLOG(3) << "Send a message from interceptor " << src_id
            << " to interceptor " << dst_id << ", which are in the same ranks.";
    return SendIntraRank(interceptor_message);
  } else {
    VLOG(3) << "Send a message from interceptor " << src_id
            << " to interceptor " << dst_id
            << ", which are in different ranks.";
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
    int retry_time = 0;  // message bus will retry sending for 10 times
    while (retry_time < 10) {
      ++retry_time;
      if (SendInterRank(interceptor_message)) {
        VLOG(3) << "Message bus sends inter rank successfully with "
                << retry_time << " times retries.";
        return true;
      }
      VLOG(3) << "Message bus sends failed, retry after 1 seconds.";
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    VLOG(3) << "Message bus sends inter rank fail after 10 times retries.";
    return false;
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "Fleet executor does not support sending message between different "
        "ranks when Paddle is compiled with npu or "
        "isn't compiled with distributed for now."));
#endif
  }
  return true;
}

void MessageBus::ListenPort() {
  if (addr_ == "") {
    LOG(INFO) << "No need listen to port since training on single card.";
    return;
  }
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
  // function keep listen the port and handle the message
  PADDLE_ENFORCE_EQ(server_.AddService(&interceptor_message_service_,
                                       brpc::SERVER_DOESNT_OWN_SERVICE),
                    0, platform::errors::Unavailable(
                           "Message bus: init brpc service error."));

  // start the server
  const char* ip_for_brpc = addr_.c_str();
  brpc::ServerOptions options;
  options.idle_timeout_sec = -1;
  int retry_times = 0;
  int interval = 100;
  while (server_.Start(ip_for_brpc, &options) != 0) {
    ++retry_times;
    LOG(INFO) << "Message bus is retring for starting brpc for " << retry_times
              << " times. And will retry after " << interval / 1000
              << " seconds.";
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
    interval += 500;
  }
  LOG(INFO) << "Message bus's listen port thread starts successful.";

  std::set<int64_t> visit;
  InterceptorMessage tmp_msg;
  tmp_msg.set_ctrl_message(true);
  for (auto pair : interceptor_id_to_rank_) {
    if (rank_to_addr_.at(pair.second) == addr_) {
      tmp_msg.set_src_id(pair.first);
    }
  }
  for (auto pair : interceptor_id_to_rank_) {
    int64_t rank = pair.second;
    if (rank_to_addr_.at(rank) == addr_) {
      continue;
    }
    tmp_msg.set_dst_id(pair.first);
    if (visit.find(rank) == visit.end()) {
      VLOG(3) << "Message bus is testing connection for rank: " << rank << ".";
      visit.insert(rank);
      while (!Send(tmp_msg)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
      VLOG(3) << "Message bus has connected to rank: " << rank << ".";
    }
  }
#else
  LOG(WARNING)
      << "Fleet executor's ListenPort() is a fake function when Paddle is "
         "compiled with npu or Paddle isn't compiled "
         "with distributed for now.";
#endif
}

bool MessageBus::IsSameRank(int64_t src_id, int64_t dst_id) {
  // -1 is sent by carrier to source interceptor
  if (src_id == -1) src_id = dst_id;

  // check whether the dst is the same rank or different rank with src
  const auto& src_rank = interceptor_id_to_rank_.find(src_id);
  const auto& dst_rank = interceptor_id_to_rank_.find(dst_id);
  PADDLE_ENFORCE_NE(
      src_rank, interceptor_id_to_rank_.end(),
      platform::errors::NotFound(
          "Cannot find rank for src interceptor id %lld. Init error.", src_id));
  PADDLE_ENFORCE_NE(
      dst_rank, interceptor_id_to_rank_.end(),
      platform::errors::NotFound(
          "Cannot find rank for dst interceptor id %lld. Init error.", dst_id));
  if (addr_ == "") {
    // single card training, must be same rank
    return true;
  }
  const auto& src_ip = rank_to_addr_.find(src_rank->second);
  PADDLE_ENFORCE_NE(src_ip, rank_to_addr_.end(),
                    platform::errors::NotFound(
                        "Cannot find addr for src rank id %lld. Init error.",
                        src_rank->second));
  PADDLE_ENFORCE_EQ(
      src_ip->second, addr_,
      platform::errors::Fatal("The src interceptor's addr is %s, while the "
                              "message bus's addr is %s, which are different. "
                              "Init error.",
                              src_ip->second, addr_));
  return src_rank->second == dst_rank->second;
}

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
bool MessageBus::SendInterRank(const InterceptorMessage& interceptor_message) {
  // send the message inter rank (dst is different rank with src)
  int64_t dst_id = interceptor_message.dst_id();
  int64_t dst_rank = interceptor_id_to_rank_[dst_id];
  auto dst_ip = rank_to_addr_.find(dst_rank);
  PADDLE_ENFORCE_NE(dst_ip, rank_to_addr_.end(),
                    platform::errors::InvalidArgument(
                        "Cannot find rank for dst interceptor id %lld. "
                        "Init error.",
                        dst_id));
  VLOG(3) << "Message bus sending to addr: " << dst_ip->second;
  const char* dst_ip_for_brpc = dst_ip->second.c_str();
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connect_timeout_ms = 1000;
  options.timeout_ms = 1000;
  options.max_retry = 5;
  PADDLE_ENFORCE_EQ(
      channel.Init(dst_ip_for_brpc, &options), 0,
      platform::errors::Unavailable("Message bus: init brpc channel error."));
  TheInterceptorMessageService_Stub stub(&channel);
  InterceptorResponse response;
  brpc::Controller ctrl;
  ctrl.set_log_id(0);
  stub.InterceptorMessageService(&ctrl, &interceptor_message, &response, NULL);
  if (!ctrl.Failed()) {
    if (response.rst()) {
      VLOG(3) << "Message bus: brpc sends success.";
      return true;
    } else {
      VLOG(4) << "Message bus: InterceptorMessageService error.";
      return false;
    }
  } else {
    VLOG(4) << "Message bus: brpc sends failed with error text: "
            << ctrl.ErrorText();
    return false;
  }
}
#endif

bool MessageBus::SendIntraRank(const InterceptorMessage& interceptor_message) {
  // send the message intra rank (dst is the same rank with src)
  return Carrier::Instance().EnqueueInterceptorMessage(interceptor_message);
}

}  // namespace distributed
}  // namespace paddle
