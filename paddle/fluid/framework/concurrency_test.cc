/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <thread>
#include <chrono>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/executor.h"

USE_NO_KERNEL_OP(go);
USE_NO_KERNEL_OP(channel_close);
USE_NO_KERNEL_OP(channel_create);
//USE_NO_KERNEL_OP(channel_recv);
USE_NO_KERNEL_OP(channel_send);
USE_NO_KERNEL_OP(elementwise_add);

namespace f = paddle::framework;
namespace p = paddle::platform;

namespace paddle {
namespace framework {

void CreateIntVariable(Scope &scope, p::CPUPlace &place, std::string name, int value) {
  // Create LoDTensor<int> of dim [1,1]
  auto var = scope.Var(name);
  auto tensor = var->GetMutable<LoDTensor>();
  tensor->Resize({1, 1});
  int *expect = tensor->mutable_data<int>(place);
  expect[0] = value;
}

void InitTensorsInScope(Scope &scope, p::CPUPlace &place) {
  p::CPUDeviceContext ctx(place);

  // Create channel variable
  scope.Var("Channel");

  // Create Variables, x0 will be put into channel,
  // result will be pulled from channel
  CreateIntVariable(scope, place, "x0", 99);
  CreateIntVariable(scope, place, "Status", 0);
  CreateIntVariable(scope, place, "result", 0);
}

void AddOp(const std::string &type, const VariableNameMap &inputs,
           const VariableNameMap &outputs, AttributeMap attrs,
           BlockDesc *block) {
  // insert op
  auto op = block->AppendOp();
  op->SetType(type);
  for (auto &kv : inputs) {
    op->SetInput(kv.first, kv.second);
  }
  for (auto &kv : outputs) {
    op->SetOutput(kv.first, kv.second);
  }
  op->SetAttrMap(attrs);
}

TEST(Concurrency, Go_Op) {
  Scope scope;
  p::CPUPlace place;

  // Initialize scope variables
  InitTensorsInScope(scope, place);

  framework::Executor executor(place);
  ProgramDesc program;
  BlockDesc *block = program.MutableBlock(0);

  // Create channel OP
  AddOp("channel_create",
        {},
        {{"Output", {"Channel"}}},
        {{"capacity", 10}, {"data_type", f::proto::VarType::LOD_TENSOR}},
        block);

  // Create Go Op routine
  ProgramDesc goOpProgram;
  BlockDesc *goOpBlock = goOpProgram.MutableBlock(0);
  AddOp("channel_send",
        {{"Channel", {"Channel"}}, {"X", {"x0"}}},
        {{"Status", {"Status"}}},
        {},
        goOpBlock);

  // Create Go Op
  AddOp("go",
        {{"X", {"Channel", "x0"}}},
        {},
        {{"sub_block", goOpBlock}},
        block);

  // Create Channel Receive Op
//  AddOp("channel_recv",
//        {{"Channel", {"Channel"}}},
//        {{"Status", {"Status"}}, {"Output", {"result"}}},
//        {},
//        block);

  // Create Channel Close Op
  AddOp("channel_close",
        {{"Channel", {"Channel"}}},
        {},
        {},
        block);

  executor.Run(program, &scope, 0, true, true);
//  std::this_thread::sleep_for(std::chrono::minutes(1));

//  const LoDTensor &tensor = (scope.FindVar("result"))->Get<LoDTensor>();
//  auto *data = tensor.data<int>();
//  EXPECT_EQ(data[0], 99);
}
}  // namespace framework
}  // namespace paddle
