// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <unordered_map>

#include "gtest/gtest.h"

#include "paddle/fluid/distributed/fleet_executor/carrier.h"
#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"

namespace paddle {
namespace distributed {

class FakeInterceptor : public Interceptor {
 public:
  FakeInterceptor(int64_t interceptor_id, TaskNode* node)
      : Interceptor(interceptor_id, node) {
    step_ = 0;
    RegisterMsgHandle([this](const InterceptorMessage& msg) { NOP(msg); });
  }

  void NOP(const InterceptorMessage& msg) {
    if (msg.message_type() == DATA_IS_READY) {
      std::cout << "FakeInterceptor run in scope " << msg.scope_idx()
                << std::endl;
      InterceptorMessage reply;
      reply.set_message_type(DATA_IS_USELESS);
      Send(SOURCE_ID, reply);
      step_++;
      if (step_ == node_->max_run_times()) {
        carrier_->WakeUp();
      }
    }
  }

 private:
  int64_t step_;
};

TEST(SourceInterceptor, Source) {
  std::string carrier_id = "0";
  Carrier* carrier =
      GlobalMap<std::string, Carrier>::Create(carrier_id, carrier_id);
  carrier->Init(0, {{SOURCE_ID, 0}, {0, 0}});

  MessageBus* msg_bus = GlobalVal<MessageBus>::Create();
  msg_bus->Init(0, {{0, "127.0.0.0:0"}}, "");

  // NOTE: don't delete, otherwise interceptor will use undefined node
  TaskNode* source =
      new TaskNode(0, SOURCE_ID, 0, 3, 0);         // role, rank, task_id
  TaskNode* node_a = new TaskNode(0, 0, 0, 3, 0);  // role, rank, task_id

  source->AddDownstreamTask(0, 1);
  node_a->AddUpstreamTask(SOURCE_ID, 1);
  carrier->SetInterceptor(
      SOURCE_ID, InterceptorFactory::Create("Source", SOURCE_ID, source));
  carrier->SetInterceptor(0, std::make_unique<FakeInterceptor>(0, node_a));

  // start
  InterceptorMessage msg;
  msg.set_message_type(START);
  msg.set_dst_id(SOURCE_ID);
  carrier->EnqueueInterceptorMessage(msg);

  carrier->Wait();
  carrier->Release();
}

}  // namespace distributed
}  // namespace paddle
