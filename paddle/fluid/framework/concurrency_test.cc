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

#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"

USE_NO_KERNEL_OP(go);
USE_NO_KERNEL_OP(channel_close);
USE_NO_KERNEL_OP(channel_create);
USE_NO_KERNEL_OP(channel_recv);
USE_NO_KERNEL_OP(channel_send);
USE_NO_KERNEL_OP(elementwise_add);
USE_NO_KERNEL_OP(select);
USE_NO_KERNEL_OP(conditional_block);
USE_NO_KERNEL_OP(equal);
USE_NO_KERNEL_OP(assign);
USE_NO_KERNEL_OP(while);
USE_NO_KERNEL_OP(print);

namespace f = paddle::framework;
namespace p = paddle::platform;

namespace paddle {
namespace framework {

template <typename T>
LoDTensor *CreateVariable(Scope *scope, const p::CPUPlace &place,
                          std::string name, T value) {
  // Create LoDTensor<int> of dim [1]
  auto var = scope->Var(name);
  auto tensor = var->GetMutable<LoDTensor>();
  tensor->Resize({1});
  T *expect = tensor->mutable_data<T>(place);
  expect[0] = value;
  return tensor;
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

void AddCase(ProgramDesc *program, Scope *scope, p::CPUPlace *place,
             BlockDesc *casesBlock, int caseId, int caseType,
             std::string caseChannel, std::string caseVarName,
             std::function<void(BlockDesc *, Scope *)> func) {
  std::string caseCondName = std::string("caseCond") + std::to_string(caseId);
  std::string caseCondXVarName =
      std::string("caseCondX") + std::to_string(caseId);

  BlockDesc *caseBlock = program->AppendBlock(*casesBlock);
  func(caseBlock, scope);

  CreateVariable(scope, *place, caseCondName, false);
  CreateVariable(scope, *place, caseCondXVarName, caseId);
  CreateVariable(scope, *place, caseVarName, caseId);

  scope->Var("step_scope");

  AddOp("equal", {{"X", {caseCondXVarName}}, {"Y", {"caseToExecute"}}},
        {{"Out", {caseCondName}}}, {}, casesBlock);

  AddOp("conditional_block", {{"X", {caseCondName}}, {"Params", {}}},
        {{"Out", {}}, {"Scope", {"step_scope"}}},
        {{"sub_block", caseBlock}, {"is_scalar_condition", true}}, casesBlock);
}

void AddFibonacciSelect(Scope *scope, p::CPUPlace *place, ProgramDesc *program,
                        BlockDesc *parentBlock, std::string dataChanName,
                        std::string quitChanName) {
  BlockDesc *whileBlock = program->AppendBlock(*parentBlock);

  CreateVariable(scope, *place, "whileExitCond", true);
  CreateVariable(scope, *place, "caseToExecute", -1);
  CreateVariable(scope, *place, "case1var", 0);

  CreateVariable(scope, *place, "xtemp", 0);

  // TODO(thuan): Need to create fibXToSend, since channel send moves the actual
  // data,
  // which causes the data to be no longer accessible to do the fib calculation
  // TODO(abhinav): Change channel send to do a copy instead of a move!
  CreateVariable(scope, *place, "fibXToSend", 0);

  CreateVariable(scope, *place, "fibX", 0);
  CreateVariable(scope, *place, "fibY", 1);
  CreateVariable(scope, *place, "quitVar", 0);

  BlockDesc *casesBlock = program->AppendBlock(*whileBlock);
  std::function<void(BlockDesc * caseBlock)> f = [](BlockDesc *caseBlock) {};

  // TODO(thuan): Remove this once we change channel send to do a copy instead
  // of move
  AddOp("assign", {{"X", {"fibX"}}}, {{"Out", {"fibXToSend"}}}, {}, whileBlock);

  // Case 0: Send to dataChanName
  std::function<void(BlockDesc * caseBlock, Scope * scope)> case0Func = [&](
      BlockDesc *caseBlock, Scope *scope) {
    AddOp("assign", {{"X", {"fibX"}}}, {{"Out", {"xtemp"}}}, {}, caseBlock);
    AddOp("assign", {{"X", {"fibY"}}}, {{"Out", {"fibX"}}}, {}, caseBlock);
    AddOp("elementwise_add", {{"X", {"xtemp"}}, {"Y", {"fibY"}}},
          {{"Out", {"fibY"}}}, {}, caseBlock);
  };
  AddCase(program, scope, place, casesBlock, 0, 1, dataChanName, "fibXToSend",
          case0Func);
  std::string case0Config =
      std::string("0,1,") + dataChanName + std::string(",fibXToSend");

  // Case 1: Receive from quitChanName
  std::function<void(BlockDesc * caseBlock, Scope * scope)> case2Func = [&](
      BlockDesc *caseBlock, Scope *scope) {
    // Exit the while loop after we receive from quit channel.
    // We assign a false to "whileExitCond" variable, which will
    // break out of while_op loop
    CreateVariable(scope, *place, "whileFalse", false);
    AddOp("assign", {{"X", {"whileFalse"}}}, {{"Out", {"whileExitCond"}}}, {},
          caseBlock);
  };
  AddCase(program, scope, place, casesBlock, 1, 2, quitChanName, "quitVar",
          case2Func);
  std::string case1Config =
      std::string("1,2,") + quitChanName + std::string(",quitVar");

  // Select block
  AddOp("select", {{"X", {dataChanName, quitChanName}},
                   {"case_to_execute", {"caseToExecute"}}},
        {{"Out", {}}},
        {{"sub_block", casesBlock},
         {"cases", std::vector<std::string>{case0Config, case1Config}}},
        whileBlock);

  scope->Var("stepScopes");
  AddOp("while",
        {{"X", {dataChanName, quitChanName}}, {"Condition", {"whileExitCond"}}},
        {{"Out", {}}, {"StepScopes", {"stepScopes"}}},
        {{"sub_block", whileBlock}}, parentBlock);
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
  CreateVariable(&scope, place, "Status", false);
  CreateVariable(&scope, place, "x0", 99);
  CreateVariable(&scope, place, "result", 0);

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

  // After we call executor.run, the Go operator should do a channel_send to
  // set the "result" variable to 99.
  auto *finalData = tensor.data<int>();
  EXPECT_EQ(finalData[0], 99);
}

/**
 * This test implements the fibonacci function using go_op and select_op
 */
TEST(Concurrency, Select) {
  Scope scope;
  p::CPUPlace place;

  // Initialize scope variables
  p::CPUDeviceContext ctx(place);

  CreateVariable(&scope, place, "Status", false);
  CreateVariable(&scope, place, "result", 0);
  CreateVariable(&scope, place, "currentXFib", 0);

  framework::Executor executor(place);
  ProgramDesc program;
  BlockDesc *block = program.MutableBlock(0);

  // Create channel OP
  std::string dataChanName = "Channel";
  scope.Var(dataChanName);
  AddOp("channel_create", {}, {{"Out", {dataChanName}}},
        {{"capacity", 0}, {"data_type", f::proto::VarType::LOD_TENSOR}}, block);

  std::string quitChanName = "Quit";
  scope.Var(quitChanName);
  AddOp("channel_create", {}, {{"Out", {quitChanName}}},
        {{"capacity", 0}, {"data_type", f::proto::VarType::LOD_TENSOR}}, block);

  // Create Go Op routine, which loops 10 times over fibonacci sequence
  CreateVariable(&scope, place, "xReceiveVar", 0);

  BlockDesc *goOpBlock = program.AppendBlock(program.Block(0));
  for (int i = 0; i < 10; ++i) {
    AddOp("channel_recv", {{"Channel", {dataChanName}}},
          {{"Status", {"Status"}}, {"Out", {"currentXFib"}}}, {}, goOpBlock);
    AddOp("print", {{"In", {"currentXFib"}}}, {{"Out", {"currentXFib"}}},
          {{"first_n", 100},
           {"summarize", -1},
           {"print_tensor_name", false},
           {"print_tensor_type", true},
           {"print_tensor_shape", false},
           {"print_tensor_lod", false},
           {"print_phase", std::string("FORWARD")},
           {"message", std::string("X: ")}},
          goOpBlock);
  }

  CreateVariable(&scope, place, "quitSignal", 0);
  AddOp("channel_send", {{"Channel", {quitChanName}}, {"X", {"quitSignal"}}},
        {{"Status", {"Status"}}}, {}, goOpBlock);

  // Create Go Op
  AddOp("go", {{"X", {dataChanName, quitChanName}}}, {},
        {{"sub_block", goOpBlock}}, block);

  AddFibonacciSelect(&scope, &place, &program, block, dataChanName,
                     quitChanName);

  // Create Channel Close Op
  AddOp("channel_close", {{"Channel", {dataChanName}}}, {}, {}, block);
  AddOp("channel_close", {{"Channel", {quitChanName}}}, {}, {}, block);

  executor.Run(program, &scope, 0, true, true);

  // After we call executor.run, "result" variable should be equal to 34
  // (which is 10 loops through fibonacci sequence)
  const LoDTensor &tensor = (scope.FindVar("currentXFib"))->Get<LoDTensor>();
  auto *finalData = tensor.data<int>();
  EXPECT_EQ(finalData[0], 34);
}

}  // namespace framework
}  // namespace paddle
