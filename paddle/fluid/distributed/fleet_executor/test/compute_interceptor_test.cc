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

#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

class StopInterceptor : public Interceptor {
 public:
  StopInterceptor(int64_t interceptor_id, TaskNode* node)
      : Interceptor(interceptor_id, node) {
    RegisterMsgHandle([this](const InterceptorMessage& msg) { Stop(msg); });
  }

  void Stop(const InterceptorMessage& msg) {
    std::cout << GetInterceptorId() << " recv msg from " << msg.src_id()
              << std::endl;
    count_ += 1;
    if (count_ == 1) return;
    InterceptorMessage stop;
    stop.set_message_type(STOP);
    Send(0, stop);
    Send(1, stop);
    Send(2, stop);
    Send(3, stop);
  }
  int count_{0};
};

class StartInterceptor : public Interceptor {
 public:
  StartInterceptor(int64_t interceptor_id, TaskNode* node)
      : Interceptor(interceptor_id, node) {
    RegisterMsgHandle([this](const InterceptorMessage& msg) { NOP(msg); });
  }

  void NOP(const InterceptorMessage& msg) {
    std::cout << GetInterceptorId() << " recv msg from " << msg.src_id()
              << std::endl;
  }
};

TEST(ComputeInterceptor, Compute) {
  MessageBus& msg_bus = MessageBus::Instance();
  msg_bus.Init({{0, 0}, {1, 0}, {2, 0}, {3, 0}}, {{0, "127.0.0.0:0"}},
               "127.0.0.0:0");

  Carrier& carrier = Carrier::Instance();

  // NOTE: don't delete, otherwise interceptor will use undefined node
  TaskNode* node_a = new TaskNode(0, 0, 0, 0, 0);  // role, rank, task_id
  TaskNode* node_b = new TaskNode(0, 0, 1, 0, 0);
  TaskNode* node_c = new TaskNode(0, 0, 2, 0, 0);
  TaskNode* node_d = new TaskNode(0, 0, 3, 0, 0);

  // a->b->c->d
  node_a->AddDownstreamTask(1);
  node_b->AddUpstreamTask(0);
  node_b->AddDownstreamTask(2);
  node_c->AddUpstreamTask(1);
  node_c->AddDownstreamTask(3);
  node_d->AddUpstreamTask(2);

  Interceptor* a =
      carrier.SetInterceptor(0, std::make_unique<StartInterceptor>(0, node_a));
  carrier.SetInterceptor(1, InterceptorFactory::Create("Compute", 1, node_b));
  carrier.SetInterceptor(2, InterceptorFactory::Create("Compute", 2, node_c));
  carrier.SetInterceptor(3, std::make_unique<StopInterceptor>(3, node_c));

  carrier.SetCreatingFlag(false);

  InterceptorMessage msg;
  msg.set_message_type(DATA_IS_READY);
  // double buff, send twice
  a->Send(1, msg);
  a->Send(1, msg);
}

}  // namespace distributed
}  // namespace paddle
