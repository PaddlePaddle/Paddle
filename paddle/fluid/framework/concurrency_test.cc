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

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"

USE_NO_KERNEL_OP(go);
USE_NO_KERNEL_OP(channel_close);
USE_NO_KERNEL_OP(channel_create);
USE_NO_KERNEL_OP(channel_recv);
USE_NO_KERNEL_OP(channel_send);
USE_NO_KERNEL_OP(elementwise_add);
USE_NO_KERNEL_OP(select);

namespace f = paddle::framework;
namespace p = paddle::platform;

namespace paddle {
namespace framework {

template <typename T>
void CreateIntVariable(Scope &scope, p::CPUPlace &place, std::string name,
                       T value) {
  // Create LoDTensor<int> of dim [1,1]
  auto var = scope.Var(name);
  auto tensor = var->GetMutable<LoDTensor>();
  tensor->Resize({1, 1});
  T *expect = tensor->mutable_data<T>(place);
  expect[0] = value;
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

void CreateSelect(Scope *scope, p::CPUPlace *place, ProgramDesc *program, BlockDesc *parentBlock) {
  CreateIntVariable(*scope, *place, "case0var", 0);
  CreateIntVariable(*scope, *place, "case1var", 0);
  CreateIntVariable(*scope, *place, "case0cond", true);

  BlockDesc *casesBlock = program->AppendBlock(*parentBlock);
  BlockDesc *case0Block = program->AppendBlock(*casesBlock);
  BlockDesc *case1Block = program->AppendBlock(*casesBlock);

  AddOp("conditional_block",
        {{"X", {"case0cond"}}},
        {},
        {{"sub_block", case0Block},
         {"is_scalar_condition", true},
         {"case_index", 0},
         {"case_type", 2},
         {"case_channel", std::string("Channel1")},
         {"case_channel_var", std::string("case0var")}},
        casesBlock);

  AddOp("conditional_block",
        {{"X", {"case0cond"}}},
        {},
        {{"sub_block", case1Block},
         {"is_scalar_condition", true},
         {"case_index", 1},
         {"case_type", 1},
         {"case_channel", std::string("Channel1")},
         {"case_channel_var", std::string("case1var")}},
        casesBlock);

  AddOp("select",
        {{"X", {"Channel1", "Channel2"}}},
        {},
        {{"sub_block", casesBlock}},
        parentBlock);
}

TEST(Concurrency, Go_Op) {
  Scope scope;
  p::CPUPlace place;

  // Initialize scope variables
  p::CPUDeviceContext ctx(place);

  // Create channel variable
  scope.Var("Channel");

  // Create Variables, x0 will be put into channel,
  // result will be pulled from channel
  CreateIntVariable(scope, place, "Status", false);
  CreateIntVariable(scope, place, "x0", 99);
  CreateIntVariable(scope, place, "result", 0);

  framework::Executor executor(place);
  ProgramDesc program;
  BlockDesc *block = program.MutableBlock(0);

  // Create channel OP
  AddOp("channel_create", {}, {{"Out", {"Channel"}}},
        {{"capacity", 10}, {"data_type", f::proto::VarType::LOD_TENSOR}},
        block);

  // Create Go Op routine
  BlockDesc *goOpBlock = program.AppendBlock(program.Block(0));
  AddOp("channel_send", {{"Channel", {"Channel"}}, {"X", {"x0"}}},
        {{"Status", {"Status"}}}, {}, goOpBlock);

  // Create Go Op
  AddOp("go", {{"X", {"Channel", "x0"}}}, {}, {{"sub_block", goOpBlock}},
        block);

  // Create Channel Receive Op
  AddOp("channel_recv", {{"Channel", {"Channel"}}},
        {{"Status", {"Status"}}, {"Out", {"result"}}}, {}, block);

  // Create Channel Close Op
  AddOp("channel_close", {{"Channel", {"Channel"}}}, {}, {}, block);

  // Check the result tensor to make sure it is set to 0
  const LoDTensor &tensor = (scope.FindVar("result"))->Get<LoDTensor>();
  auto *initialData = tensor.data<int>();
  EXPECT_EQ(initialData[0], 0);

  executor.Run(program, &scope, 0, true, true);

  // After we call executor.run, the Go operator should do a channel_send to set
  // the
  // "result" variable to 99
  auto *finalData = tensor.data<int>();
  EXPECT_EQ(finalData[0], 99);
}

TEST(Concurrency, Select) {
  Scope scope;
  p::CPUPlace place;

  // Initialize scope variables
  // Initialize scope variables
  p::CPUDeviceContext ctx(place);

  // Create channels variable
  scope.Var("Channel1");
  scope.Var("Channel2");

  // Create Variables, x0 will be put into channel,
  // result will be pulled from channel
  CreateIntVariable(scope, place, "Status", false);
  CreateIntVariable(scope, place, "x0", 99);
  CreateIntVariable(scope, place, "result", 0);

  framework::Executor executor(place);
  ProgramDesc program;
  BlockDesc *block = program.MutableBlock(0);

  // Create channel OP
  AddOp("channel_create", {}, {{"Out", {"Channel1"}}},
        {{"capacity", 10}, {"data_type", f::proto::VarType::LOD_TENSOR}},
        block);

  CreateSelect(&scope, &place, &program, block);

//  // Create Go Op routine
//  BlockDesc *goOpBlock = program.AppendBlock(program.Block(0));
//  AddOp("channel_send", {{"Channel", {"Channel"}}, {"X", {"x0"}}},
//        {{"Status", {"Status"}}}, {}, goOpBlock);
//
//  // Create Go Op
//  AddOp("go", {{"X", {"Channel", "x0"}}}, {}, {{"sub_block", goOpBlock}},
//        block);

  // Create Channel Receive Op
  AddOp("channel_recv", {{"Channel", {"Channel"}}},
        {{"Status", {"Status"}}, {"Out", {"result"}}}, {}, block);

  // Create Channel Close Op
  AddOp("channel_close", {{"Channel", {"Channel"}}}, {}, {}, block);

//  // Check the result tensor to make sure it is set to 0
//  const LoDTensor &tensor = (scope.FindVar("result"))->Get<LoDTensor>();
//  auto *initialData = tensor.data<int>();
//  EXPECT_EQ(initialData[0], 0);

  executor.Run(program, &scope, 0, true, true);

  // After we call executor.run, the Go operator should do a channel_send to set
  // the
  // "result" variable to 99
//  auto *finalData = tensor.data<int>();
//  EXPECT_EQ(finalData[0], 99);
}

}  // namespace framework
}  // namespace paddle
