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
#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"

namespace paddle {
namespace distributed {

void MessageBus::Init(
    int64_t rank, const std::unordered_map<int64_t, std::string>& rank_to_addr,
    const std::string& addr) {
  PADDLE_ENFORCE_EQ(is_init_, false, platform::errors::AlreadyExists(
                                         "MessageBus is already init."));
  rank_ = rank;
  is_init_ = true;
  rank_to_addr_ = rank_to_addr;
  addr_ = addr;

  if (addr_ != "") {
    const auto& addr = GetAddr(rank_);
    PADDLE_ENFORCE_EQ(addr, addr_,
                      platform::errors::Fatal(
                          "The current rank's addr is %s, while the "
                          "message bus's addr is %s, which are different. "
                          "Init error.",
                          addr, addr_));
  }

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
  VLOG(3) << "Message bus releases resource.";
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  server_.Stop(1000);
  server_.Join();
#endif
}

const std::string& MessageBus::GetAddr(int64_t rank) const {
  PADDLE_ENFORCE_NE(
      rank_to_addr_.find(rank), rank_to_addr_.end(),
      platform::errors::NotFound("Cannot find addr rank id %lld.", rank));
  return rank_to_addr_.at(rank);
}

bool MessageBus::Send(int64_t dst_rank,
                      const InterceptorMessage& interceptor_message) {
  PADDLE_ENFORCE_EQ(
      IsInit(), true,
      platform::errors::PreconditionNotMet(
          "Using message bus since it has not been initialized."));
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  int retry_time = 0;  // message bus will retry sending for 10 times
  while (retry_time < 10) {
    ++retry_time;
    if (SendInterRank(dst_rank, interceptor_message)) {
      VLOG(3) << "Message bus sends inter rank successfully with " << retry_time
              << " times retries.";
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
  return true;
}

void MessageBus::IncreaseBarrierCount() {
  VLOG(3) << "IncreaseBarrierCount";
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ++count_;
    cv_.notify_one();
  }
  VLOG(3) << "End IncreaseBarrierCount";
}

void MessageBus::Barrier() {
  // gather to root
  if (rank_ != 0) {
    InterceptorMessage ctrl_msg;
    ctrl_msg.set_ctrl_message(true);
    ctrl_msg.set_src_id(rank_);
    ctrl_msg.set_dst_id(0);
    VLOG(3) << "Barrier Gather ctrl message from " << rank_ << " to 0";
    while (!Send(0, ctrl_msg)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
  } else {
    VLOG(3) << "Barrier 0 wait others rank ready";
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] {
      return count_ == static_cast<int>(rank_to_addr_.size() - 1);
    });
    count_ = 0;
  }

  // scatter from root
  if (rank_ == 0) {
    for (int i = 1; i < static_cast<int>(rank_to_addr_.size()); ++i) {
      InterceptorMessage ctrl_msg;
      ctrl_msg.set_ctrl_message(true);
      ctrl_msg.set_src_id(0);
      ctrl_msg.set_dst_id(i);
      VLOG(3) << "Barrier Scatter ctrl message from 0 to " << i;
      while (!Send(i, ctrl_msg)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      }
    }
  } else {
    VLOG(3) << "Barrier " << rank_ << " wait others rank ready";
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return count_ == 1; });
    count_ = 0;
  }
}

bool MessageBus::DispatchMsgToCarrier(
    const InterceptorMessage& interceptor_message) {
  const std::string& carrier_id = *GlobalVal<std::string>::Get();
  return GlobalMap<std::string, Carrier>::Get(carrier_id)
      ->EnqueueInterceptorMessage(interceptor_message);
}

void MessageBus::ListenPort() {
  if (addr_ == "") {
    LOG(INFO) << "No need listen to port since training on single card.";
    return;
  }
#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
  // function keep listen the port and handle the message
  PADDLE_ENFORCE_EQ(
      server_.AddService(&message_service_, brpc::SERVER_DOESNT_OWN_SERVICE), 0,
      platform::errors::Unavailable("Message bus: init brpc service error."));

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
#else
  LOG(WARNING)
      << "Fleet executor's ListenPort() is a fake function when Paddle is "
         "compiled with npu or Paddle isn't compiled "
         "with distributed for now.";
#endif
}

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE)
bool MessageBus::SendInterRank(int64_t dst_rank,
                               const InterceptorMessage& interceptor_message) {
  const auto& dst_addr = GetAddr(dst_rank);
  VLOG(3) << "Message bus sending to addr: " << dst_addr;
  const char* dst_addr_for_brpc = dst_addr.c_str();
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.connect_timeout_ms = 1000;
  options.timeout_ms = 1000;
  options.max_retry = 5;
  PADDLE_ENFORCE_EQ(
      channel.Init(dst_addr_for_brpc, &options), 0,
      platform::errors::Unavailable("Message bus: init brpc channel error."));
  MessageService_Stub stub(&channel);
  InterceptorResponse response;
  brpc::Controller ctrl;
  ctrl.set_log_id(0);
  if (interceptor_message.ctrl_message()) {
    stub.IncreaseBarrierCount(&ctrl, &interceptor_message, &response, NULL);
  } else {
    stub.ReceiveInterceptorMessage(&ctrl, &interceptor_message, &response,
                                   NULL);
  }
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

}  // namespace distributed
}  // namespace paddle
