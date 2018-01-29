/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <unistd.h>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/operator.h"
#include "paddle/framework/program_desc.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/selected_rows_functor.h"
#include "paddle/string/printf.h"

USE_NO_KERNEL_OP(send);
USE_NO_KERNEL_OP(listen_and_serv);
USE_OP(sum);

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

// global for simplicity.
std::unique_ptr<f::OperatorBase> listen_and_serv_op;

void InitTensorsInScope(f::Scope &scope, p::CPUPlace &place) {
  p::CPUDeviceContext ctx(place);
  for (int i = 0; i < 2; ++i) {
    auto var_name = paddle::string::Sprintf("x%d", i);
    auto var = scope.Var(var_name);
    auto tensor = var->GetMutable<f::LoDTensor>();
    tensor->Resize({10, 10});
    float *expect = tensor->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }

  auto out_var = scope.Var("Out");
  auto out_tensor = out_var->GetMutable<f::LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate
}

void InitSelectedRowsInScope(f::Scope &scope, p::CPUPlace &place) {
  p::CPUDeviceContext ctx(place);
  int64_t height = 10;
  int64_t row_numel = 10;
  m::SetConstant<p::CPUDeviceContext, float> set_one;
  // init x0
  std::vector<int64_t> rows0{0, 4, 7};
  auto x0_var = scope.Var("x0");
  auto x0 = x0_var->GetMutable<f::SelectedRows>();
  x0->set_rows(rows0);
  x0->set_height(height);
  auto x0_value = x0->mutable_value();
  x0_value->mutable_data<float>(
      f::make_ddim({static_cast<int64_t>(rows0.size()), row_numel}), place);
  set_one(ctx, x0_value, 1.0);

  // init x1
  std::vector<int64_t> rows1{2, 9};
  auto x1_var = scope.Var("x1");
  auto x1 = x1_var->GetMutable<f::SelectedRows>();
  x1->set_rows(rows1);
  x1->set_height(height);
  auto x1_value = x1->mutable_value();
  x1_value->mutable_data<float>(
      f::make_ddim({static_cast<int64_t>(rows1.size()), row_numel}), place);
  set_one(ctx, x1_value, 1.0);

  auto out_var = scope.Var("Out");
  auto out = out_var->GetMutable<f::SelectedRows>();
  auto out_value = out->mutable_value();
  out->set_height(height);
  out_value->mutable_data<float>(f::make_ddim({5, 10}), place);
}

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           f::BlockDesc *block) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(f::proto::DataType::FP32);
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

void StartServerNet(bool is_sparse) {
  f::Scope scope;
  p::CPUPlace place;
  if (is_sparse) {
    InitSelectedRowsInScope(scope, place);
  } else {
    InitTensorsInScope(scope, place);
  }

  // sub program run in listen_and_serv_op, for simple test we use sum
  f::ProgramDesc program;
  f::BlockDesc *block = program.MutableBlock(0);
  // X for server side tensors, RX for received tensers, must be of same shape.
  AddOp("sum", {{"X", {"x0", "x1"}}}, {{"Out", {"Out"}}}, {}, block);

  f::AttributeMap attrs;
  attrs.insert({"endpoint", std::string("127.0.0.1:6174")});
  attrs.insert({"ParamList", std::vector<std::string>({"Out"})});
  attrs.insert({"GradList", std::vector<std::string>({"x1"})});
  attrs.insert({"OptimizeBlock", block});
  listen_and_serv_op =
      f::OpRegistry::CreateOp("listen_and_serv", {}, {}, attrs);
  listen_and_serv_op->Run(scope, place);
}

TEST(SendRecvOp, CPUDense) {
  std::thread server_thread(StartServerNet, false);
  sleep(10);  // wait server to start
  // local net
  f::Scope scope;
  p::CPUPlace place;
  InitTensorsInScope(scope, place);

  f::AttributeMap attrs;
  attrs.insert({"endpoints", std::vector<std::string>({"127.0.0.1:6174"})});
  attrs.insert({"epmap", std::vector<std::string>({"127.0.0.1:6174"})});
  auto send_op = f::OpRegistry::CreateOp("send", {{"X", {"x1"}}},
                                         {{"Out", {"Out"}}}, attrs);
  send_op->Run(scope, place);

  auto in_var = scope.Var("x1");
  auto tensor = in_var->GetMutable<f::LoDTensor>();
  float *expected = tensor->data<float>();
  auto out_var = scope.Var("Out");
  auto target = out_var->GetMutable<f::LoDTensor>();
  // x1 * 2 == x0
  EXPECT_NE(target->memory_size(), size_t(0));
  float *actual = target->data<float>();
  for (int64_t i = 0; i < target->numel(); ++i) {
    EXPECT_EQ(expected[i] * 2, actual[i]);
  }
  listen_and_serv_op->Stop();
  server_thread.join();
  listen_and_serv_op.reset(nullptr);
}

TEST(SendRecvOp, CPUSparse) {
  std::thread server_thread(StartServerNet, true);
  sleep(3);  // wait server to start
  // local net
  f::Scope scope;
  p::CPUPlace place;
  p::CPUDeviceContext ctx(place);
  InitSelectedRowsInScope(scope, place);
  f::AttributeMap attrs;
  attrs.insert({"endpoints", std::vector<std::string>({"127.0.0.1:6174"})});
  attrs.insert({"epmap", std::vector<std::string>({"127.0.0.1:6174"})});
  auto send_op = f::OpRegistry::CreateOp("send", {{"X", {"x1"}}},
                                         {{"Out", {"Out"}}}, attrs);
  send_op->Run(scope, place);

  auto x0 = scope.Var("x0")->GetMutable<f::SelectedRows>();
  auto x1 = scope.Var("x1")->GetMutable<f::SelectedRows>();
  auto out = scope.Var("Out")->GetMutable<f::SelectedRows>();
  auto actual = out->mutable_value();

  std::unique_ptr<f::SelectedRows> expect{new f::SelectedRows()};
  auto expect_value = expect->mutable_value();
  expect_value->mutable_data<float>(f::make_ddim({5, 10}), place);

  m::SelectedRowsAdd<p::CPUDeviceContext, float> add_functor;
  add_functor(ctx, *x0, *x1, expect.get());

  EXPECT_EQ(actual->numel(), expect_value->numel());
  EXPECT_EQ(out->rows().size(), x0->rows().size() + x1->rows().size());

  for (int64_t i = 0; i < expect_value->numel(); ++i) {
    EXPECT_EQ(expect_value->mutable_data<float>(place)[i],
              actual->mutable_data<float>(place)[i]);
  }
  listen_and_serv_op->Stop();
  server_thread.join();
  listen_and_serv_op.reset();
}
