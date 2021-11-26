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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace distributed {

TEST(ComputeInterceptor, Compute) {
  MessageBus& msg_bus = MessageBus::Instance();
  msg_bus.Init({{0, 0}, {1, 0}}, {{0, "127.0.0.0:0"}}, "127.0.0.0:0");

  Carrier& carrier = Carrier::Instance();

  // NOTE: don't delete, otherwise interceptor will use undefined node
  TaskNode* node_a = new TaskNode(0, 0, 0, 2, 2);  // role, rank, task_id
  TaskNode* node_b = new TaskNode(0, 0, 1, 0, 0);

  // a->b
  node_a->AddDownstreamTask(1);
  node_b->AddUpstreamTask(0);

  Interceptor* a = carrier.SetInterceptor(
      0, InterceptorFactory::Create("Compute", 0, node_a));
  carrier.SetInterceptor(1, InterceptorFactory::Create("Compute", 1, node_b));

  carrier.SetCreatingFlag(false);

  // start
  InterceptorMessage msg;
  msg.set_message_type(DATA_IS_READY);
  msg.set_src_id(-1);
  msg.set_dst_id(0);
  carrier.EnqueueInterceptorMessage(msg);

  // stop
  InterceptorMessage stop;
  msg.set_message_type(STOP);
  msg.set_src_id(-1);
  msg.set_dst_id(0);
  carrier.EnqueueInterceptorMessage(stop);
}

}  // namespace distributed
}  // namespace paddle
