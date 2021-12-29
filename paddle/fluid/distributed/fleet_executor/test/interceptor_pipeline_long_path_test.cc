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

void LinkNodes(const std::vector<TaskNode*>& nodes) {
  size_t size = nodes.size();
  if (size <= 1) return;

  {  // i = 0
    TaskNode* now = nodes[0];
    TaskNode* next = nodes[1];
    now->AddDownstreamTask(next->task_id());
  }
  {  // i = size - 1
    TaskNode* prev = nodes[size - 2];
    TaskNode* now = nodes[size - 1];
    now->AddUpstreamTask(prev->task_id());
  }

  for (size_t i = 1; i < size - 1; ++i) {
    TaskNode* prev = nodes[i - 1];
    TaskNode* now = nodes[i];
    TaskNode* next = nodes[i + 1];

    now->AddUpstreamTask(prev->task_id());
    now->AddDownstreamTask(next->task_id());
  }
}

TEST(AmplifierInterceptor, Amplifier) {
  Carrier carrier(0, {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}});
  auto msg_bus = std::make_shared<MessageBus>();
  msg_bus->Init(0, {{0, "127.0.0.0:0"}}, "127.0.0.0:0");
  carrier.SetMsgBus(msg_bus);

  int64_t micro_steps = 3;

  // NOTE: don't delete, otherwise interceptor will use undefined node
  TaskNode* node_a = new TaskNode(0, 0, 0, 1, 0);  // role, rank, task_id
  TaskNode* node_b = new TaskNode(0, 0, 1, 1, 0);
  TaskNode* node_c = new TaskNode(0, 0, 2, 1, 0);
  TaskNode* node_d = new TaskNode(0, 0, 3, 1, 0);
  TaskNode* node_e = new TaskNode(0, 0, 4, 1, 0);
  TaskNode* node_f = new TaskNode(0, 0, 5, 1, 0);

  // a->b->c->d->e->f
  LinkNodes({node_a, node_b, node_c, node_d, node_e, node_f});

  // LR->b(1:3)->F->B->e(3:1)->U
  node_b->SetReplyUpPerSteps(micro_steps);
  node_e->SetSendDownPerSteps(micro_steps);

  carrier.SetInterceptor(0, InterceptorFactory::Create("Compute", 0, node_a));
  carrier.SetInterceptor(1, InterceptorFactory::Create("Amplifier", 1, node_b));
  carrier.SetInterceptor(2, InterceptorFactory::Create("Compute", 2, node_c));
  carrier.SetInterceptor(3, InterceptorFactory::Create("Compute", 3, node_d));
  carrier.SetInterceptor(4, InterceptorFactory::Create("Amplifier", 4, node_e));
  carrier.SetInterceptor(5, InterceptorFactory::Create("Compute", 5, node_f));

  // start
  InterceptorMessage msg;
  msg.set_message_type(DATA_IS_READY);
  msg.set_src_id(-1);
  msg.set_dst_id(0);
  carrier.EnqueueInterceptorMessage(msg);
  carrier.Wait();
  carrier.Release();
}

}  // namespace distributed
}  // namespace paddle
