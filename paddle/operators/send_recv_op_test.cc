/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// TODO(typhoonzero): add python bindings for this test as
// a RemoteOptimizer.

#include <unistd.h>
#include <thread>

#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"

USE_NO_KERNEL_OP(send);
USE_NO_KERNEL_OP(recv);
USE_OP(sum);

// global for simplicity.
std::unique_ptr<paddle::framework::OperatorBase> recv_op;

void InitTensorsInScope(paddle::framework::Scope &scope,
                        paddle::platform::CPUPlace &place) {
  paddle::platform::CPUDeviceContext ctx(place);
  auto var = scope.Var("X");
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  tensor->Resize({10, 10});
  float *expect = tensor->mutable_data<float>(place);
  for (int64_t i = 0; i < tensor->numel(); ++i) {
    expect[i] = static_cast<float>(i);
  }

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<paddle::framework::LoDTensor>();
  out_tensor->Resize({10, 10});
  tensor->mutable_data<float>(place);  // allocate
}

void AddOp(const std::string &type,
           const paddle::framework::VariableNameMap &inputs,
           const paddle::framework::VariableNameMap &outputs,
           paddle::framework::AttributeMap attrs,
           paddle::framework::BlockDescBind *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(paddle::framework::DataType::FP32);
    }
  }

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

void StartServerNet() {
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;
  InitTensorsInScope(scope, place);

  // sub program run in recv_op, for simple test we use sum
  paddle::framework::ProgramDescBind program;
  paddle::framework::BlockDescBind *block = program.MutableBlock(0);
  // X for server side tensors, RX for received tensers, must be of same shape.
  AddOp("sum", {{"X", {"X", "RX"}}}, {{"Out", {"Out"}}}, {}, block);

  paddle::framework::AttributeMap attrs;
  attrs.insert({"endpoint", std::string("127.0.0.1:6174")});
  std::string program_proto;
  PADDLE_ENFORCE(program.Proto()->SerializeToString(&program_proto));

  attrs.insert({"OptimizeProgram", program_proto});
  recv_op = paddle::framework::OpRegistry::CreateOp("recv", {{"RX", {"RX"}}},
                                                    {{"Out", {"Out"}}}, attrs);
  paddle::platform::CPUDeviceContext ctx(place);
  recv_op->Run(scope, ctx);
}

TEST(SendRecvOp, CPU) {
  std::thread server_thread(StartServerNet);
  sleep(5);  // wait server to start
  // local net
  paddle::framework::Scope scope;
  paddle::platform::CPUPlace place;
  InitTensorsInScope(scope, place);

  paddle::framework::AttributeMap attrs;
  attrs.insert({"endpoint", std::string("127.0.0.1:6174")});

  auto send_op = paddle::framework::OpRegistry::CreateOp(
      "send", {{"X", {"X"}}}, {{"Out", {"Out"}}}, attrs);
  paddle::platform::CPUDeviceContext ctx(place);
  send_op->Run(scope, ctx);

  auto in_var = scope.Var("X");
  auto tensor = in_var->GetMutable<paddle::framework::LoDTensor>();
  float *expected = tensor->data<float>();

  auto out_var = scope.Var("Out");
  auto target = out_var->GetMutable<paddle::framework::LoDTensor>();
  // send fail cause output is none.
  EXPECT_NE(target->memory_size(), size_t(0));
  float *actual = target->data<float>();
  for (int64_t i = 0; i < target->numel(); ++i) {
    EXPECT_EQ(expected[i] * 2, actual[i]);
  }
  recv_op.reset();  // dtor can shutdown and join server thread.
  server_thread.join();
}
