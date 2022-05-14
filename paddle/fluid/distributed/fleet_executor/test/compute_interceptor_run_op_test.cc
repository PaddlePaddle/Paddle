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
#include "paddle/fluid/distributed/fleet_executor/global.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/message_bus.h"
#include "paddle/fluid/distributed/fleet_executor/task_node.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP_ITSELF(elementwise_add);
USE_OP_ITSELF(fill_constant);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(full, CPU, ALL_LAYOUT);

namespace paddle {
namespace distributed {

std::vector<framework::OperatorBase*> GetOps() {
  framework::AttributeMap attrs;
  attrs["dtype"] = framework::proto::VarType::FP32;
  attrs["shape"] = phi::vectorize<int>({2, 3});
  attrs["value"] = 1.0f;

  auto zero_op = framework::OpRegistry::CreateOp("fill_constant", {},
                                                 {{"Out", {"x"}}}, attrs);

  auto op = framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"x"}}, {"Y", {"x"}}}, {{"Out", {"out"}}},
      framework::AttributeMap());

  // NOTE: don't delete
  return {zero_op.release(), op.release()};
}

framework::Scope* GetScope() {
  framework::Scope* scope = new framework::Scope();

  scope->Var("x")->GetMutable<framework::LoDTensor>();
  scope->Var("out")->GetMutable<framework::LoDTensor>();
  return scope;
}

TEST(ComputeInterceptor, Compute) {
  std::vector<framework::OperatorBase*> ops = GetOps();
  framework::Scope* scope = GetScope();
  std::vector<framework::Scope*> scopes = {scope, scope};
  platform::Place place = platform::CPUPlace();

  std::string carrier_id = "0";
  Carrier* carrier =
      GlobalMap<std::string, Carrier>::Create(carrier_id, carrier_id);
  carrier->Init(0, {{SOURCE_ID, 0}, {0, 0}, {1, 0}, {SINK_ID, 0}});

  MessageBus* msg_bus = GlobalVal<MessageBus>::Create();
  msg_bus->Init(0, {{0, "127.0.0.0:0"}}, "");

  // FIXME: don't delete, otherwise interceptor will use undefined node
  TaskNode* source =
      new TaskNode(0, SOURCE_ID, 2);  // rank, task_id, max_run_times
  TaskNode* node_a =
      new TaskNode(0, ops, 0, 0, 2, 0);  // role, ops, rank, task_id
  TaskNode* node_b = new TaskNode(0, 0, 1, 2, 0);
  TaskNode* sink = new TaskNode(0, SINK_ID, 2);

  // source->a->b->sink
  source->AddDownstreamTask(0);
  node_a->AddUpstreamTask(SOURCE_ID);
  node_a->AddDownstreamTask(1);
  node_b->AddUpstreamTask(0);
  sink->AddUpstreamTask(1);
  node_b->AddDownstreamTask(SINK_ID);

  carrier->SetInterceptor(
      SOURCE_ID, InterceptorFactory::Create("Source", SOURCE_ID, source));
  auto* a = carrier->SetInterceptor(
      0, InterceptorFactory::Create("Compute", 0, node_a));
  carrier->SetInterceptor(1, InterceptorFactory::Create("Compute", 1, node_b));
  carrier->SetInterceptor(SINK_ID,
                          InterceptorFactory::Create("Sink", SINK_ID, sink));

  a->SetPlace(place);
  a->SetMicroBatchScope(scopes);

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
