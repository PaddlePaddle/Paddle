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

#include <sys/socket.h>
#include <time.h>
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
    if (msg.message_type() == STOP) {
      stop_ = true;
      return;
    }
    std::cout << GetInterceptorId() << " recv msg, count=" << count_
              << std::endl;
    ++count_;
    if (count_ == 20 && GetInterceptorId() == 0) {
      InterceptorMessage stop;
      stop.set_message_type(STOP);
      Send(0, stop);
      Send(1, stop);
      return;
    }

    InterceptorMessage resp;
    int64_t dst = GetInterceptorId() == 0 ? 1 : 0;
    Send(dst, resp);
  }

 private:
  int count_{0};
};

REGISTER_INTERCEPTOR(PingPong, PingPongInterceptor);

TEST(InterceptorTest, PingPong) {
  std::cout << "Ping pong test through brpc" << std::endl;
  unsigned int seed = time(0);
  // random generated two ports in from 6000 to 9000
  int port0 = 6000 + rand_r(&seed) % 3000;
  int port1 = port0 + 1;

  // using socket to check the availability of the port
  int server_fd = -1;
  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  int opt = 1;
  linger ling;
  ling.l_onoff = 1;
  ling.l_linger = 0;
  setsockopt(server_fd, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  struct sockaddr_in address;
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port0);
  while (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == -1) {
    port0++;
    address.sin_port = htons(port0);
  }
  close(server_fd);

  // use another socket to check another port
  server_fd = socket(AF_INET, SOCK_STREAM, 0);
  setsockopt(server_fd, SOL_SOCKET, SO_LINGER, &ling, sizeof(ling));
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  port1 = port0 + 1;
  address.sin_port = htons(port1);
  while (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) == -1) {
    port1++;
    address.sin_port = htons(port1);
  }
  close(server_fd);

  std::string ip0 = "127.0.0.1:" + std::to_string(port0);
  std::string ip1 = "127.0.0.1:" + std::to_string(port1);
  std::cout << "ip0: " << ip0 << std::endl;
  std::cout << "ip1: " << ip1 << std::endl;

  int pid = fork();
  if (pid == 0) {
    MessageBus& msg_bus = MessageBus::Instance();
    msg_bus.Init({{0, 0}, {1, 1}}, {{0, ip0}, {1, ip1}}, ip0);

    Carrier& carrier = Carrier::Instance();

    Interceptor* a = carrier.SetInterceptor(
        0, InterceptorFactory::Create("PingPong", 0, nullptr));
    carrier.SetCreatingFlag(false);

    InterceptorMessage msg;
    a->Send(1, msg);
  } else {
    MessageBus& msg_bus = MessageBus::Instance();
    msg_bus.Init({{0, 0}, {1, 1}}, {{0, ip0}, {1, ip1}}, ip1);

    Carrier& carrier = Carrier::Instance();

    carrier.SetInterceptor(1,
                           InterceptorFactory::Create("PingPong", 1, nullptr));
    carrier.SetCreatingFlag(false);
  }
}

}  // namespace distributed
}  // namespace paddle
