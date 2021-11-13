/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"

namespace paddle {
namespace distributed {

class PingPongInterceptor : public Interceptor {
 public:
  PingPongInterceptor(int64_t interceptor_id, TaskNode* node)
      : Interceptor(interceptor_id, node) {
    RegisterMsgHandle([this](const InterceptorMessage& msg) { PingPong(msg); });
  }

  void PingPong(const InterceptorMessage& msg) {
    std::cout << GetInterceptorId() << " recv msg, count=" << count_
              << std::endl;
    ++count_;
    if (count_ == 20) {
      InterceptorMessage stop;
      stop.set_message_type(STOP);
      Send(GetInterceptorId() == 0 ? 1 : 0, stop);
      return;
    }

    InterceptorMessage resp;
    Send(msg.src_id(), resp);
  }

 private:
  int count_{0};
};

#if defined(PADDLE_WITH_DISTRIBUTE) && defined(PADDLE_WITH_PSCORE) && \
    !defined(PADDLE_WITH_ASCEND_CL)
TEST(InterceptorTestRemote, RemotePingPong) {
  std::cout << "Test ping pong through brpc.";
  pid_t pid = fork();
  if (pid < 0) {
    std::cout << "Fork error, exit remote ping pong test." << std::endl;
  } else if (pid == 0) {
    MessageBus& msg_bus = MessageBus::Instance();
    msg_bus.Init({{0, 0}, {1, 1}},
                 {{0, "127.0.0.1:6912"}, {1, "127.0.0.1:7950"}},
                 "127.0.0.1:7950");

    Carrier& carrier = Carrier::Instance();

    Interceptor* a = carrier.SetInterceptor(
        1, std::make_unique<PingPongInterceptor>(1, nullptr));

    InterceptorMessage msg;
    while (!a->Send(0, msg)) {
    }
  } else {
    MessageBus& msg_bus = MessageBus::Instance();
    msg_bus.Init({{0, 0}, {1, 1}},
                 {{0, "127.0.0.1:6912"}, {1, "127.0.0.1:7950"}},
                 "127.0.0.1:6912");

    Carrier& carrier = Carrier::Instance();

    Interceptor* a = carrier.SetInterceptor(
        0, std::make_unique<PingPongInterceptor>(0, nullptr));

    InterceptorMessage msg;
    while (!a->Send(1, msg)) {
    }
  }
}
#endif

}  // namespace distributed
}  // namespace paddle
