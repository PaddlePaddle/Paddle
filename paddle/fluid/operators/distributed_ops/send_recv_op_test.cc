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

#include <unistd.h>
#include <string>
#include <thread>  // NOLINT

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/distributed_ops/listen_and_serv_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/string/printf.h"

USE_NO_KERNEL_OP(send);
USE_NO_KERNEL_OP(listen_and_serv);
USE_OP(sum);

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;
namespace d = paddle::operators::distributed

    // global for simplicity.
    std::unique_ptr<f::OperatorBase>
        listen_and_serv_op;
int selected_port;

void InitTensorsInScope(const p::CPUPlace &place, f::Scope *scope) {
  p::CPUDeviceContext ctx(place);
  for (int i = 0; i < 2; ++i) {
    auto var_name = paddle::string::Sprintf("x%d", i);
    auto var = scope->Var(var_name);
    auto tensor = var->GetMutable<f::LoDTensor>();
    tensor->Resize({10, 10});
    float *expect = tensor->mutable_data<float>(place);
    for (int64_t i = 0; i < tensor->numel(); ++i) {
      expect[i] = static_cast<float>(i);
    }
  }

  auto out_var = scope->Var("Out");
  auto out_tensor = out_var->GetMutable<f::LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate
}

void InitSelectedRowsInScope(const p::CPUPlace &place, f::Scope *scope) {
  p::CPUDeviceContext ctx(place);
  int64_t height = 10;
  int64_t row_numel = 10;
  m::SetConstant<p::CPUDeviceContext, float> set_one;
  // init x0
  std::vector<int64_t> rows0{0, 4, 7};
  auto x0_var = scope->Var("x0");
  auto x0 = x0_var->GetMutable<f::SelectedRows>();
  x0->set_rows(rows0);
  x0->set_height(height);
  auto x0_value = x0->mutable_value();
  x0_value->mutable_data<float>(
      f::make_ddim({static_cast<int64_t>(rows0.size()), row_numel}), place);
  set_one(ctx, x0_value, 1.0);

  // init x1
  std::vector<int64_t> rows1{2, 9};
  auto x1_var = scope->Var("x1");
  auto x1 = x1_var->GetMutable<f::SelectedRows>();
  x1->set_rows(rows1);
  x1->set_height(height);
  auto x1_value = x1->mutable_value();
  x1_value->mutable_data<float>(
      f::make_ddim({static_cast<int64_t>(rows1.size()), row_numel}), place);
  set_one(ctx, x1_value, 1.0);

  auto out_var = scope->Var("Out");
  auto out = out_var->GetMutable<f::SelectedRows>();
  auto out_value = out->mutable_value();
  out->set_height(height);
  out_value->mutable_data<float>(f::make_ddim({5, 10}), place);
}

void AddOp(const std::string &type, const f::VariableNameMap &inputs,
           const f::VariableNameMap &outputs, f::AttributeMap attrs,
           f::BlockDesc *block, bool is_sparse) {
  // insert output
  for (auto kv : outputs) {
    for (auto v : kv.second) {
      auto var = block->Var(v);
      var->SetDataType(f::proto::VarType::FP32);
      var->SetPersistable(true);
      if (is_sparse) {
        var->SetType(f::proto::VarType::SELECTED_ROWS);
      }
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

void StartServerNet(bool is_sparse, std::atomic<bool> *initialized) {
  f::Scope scope;
  p::CPUPlace place;
  VLOG(4) << "before init tensor";
  if (is_sparse) {
    InitSelectedRowsInScope(place, &scope);
  } else {
    InitTensorsInScope(place, &scope);
  }
  // sub program run in listen_and_serv_op, for simple test we use sum
  f::ProgramDesc program;
  const auto &root_block = program.Block(0);
  std::vector<framework::BlockDesc *> optimize_blocks;
  auto *optimize_block = program.AppendBlock(root_block);
  optimize_blocks.push_back(optimize_block);

  auto *prefetch_block = program.AppendBlock(root_block);
  // X for server side tensors, RX for received tensors, must be of same shape.
  AddOp("sum", {{"X", {"x0", "x1"}}}, {{"Out", {"Out"}}}, {}, optimize_block,
        is_sparse);
  f::AttributeMap attrs;
  attrs.insert({"endpoint", std::string("127.0.0.1:0")});
  attrs.insert({"Fanin", 1});
  attrs.insert({"ParamList", std::vector<std::string>({"Out"})});
  attrs.insert({"GradList", std::vector<std::string>({"x1"})});
  attrs.insert({"optimize_blocks", optimize_blocks});
  attrs.insert({"PrefetchBlock", prefetch_block});
  attrs.insert({"grad_to_block_id", std::vector<std::string>({""})});
  attrs.insert({"distributed_mode", d::DistributedMode::kSync});
  VLOG(4) << "before init op";
  listen_and_serv_op =
      f::OpRegistry::CreateOp("listen_and_serv", {{"X", {"x1"}}}, {}, attrs);
  *initialized = true;
  listen_and_serv_op->Run(scope, place);
  LOG(INFO) << "server exit";
}

TEST(SendRecvOp, CPUDense) {
  std::atomic<bool> initialized{false};
  std::thread server_thread(StartServerNet, false, &initialized);
  while (!initialized) {
  }

  static_cast<paddle::operators::ListenAndServOp *>(listen_and_serv_op.get())
      ->WaitServerReady();

  // local net
  f::Scope scope;
  p::CPUPlace place;
  InitTensorsInScope(place, &scope);
  // create rpc client var
  scope.Var("RPC_CLIENT_VAR");

  f::AttributeMap attrs;
  auto *listen_and_serv_op_ptr =
      static_cast<paddle::operators::ListenAndServOp *>(
          listen_and_serv_op.get());
  ASSERT_TRUE(listen_and_serv_op_ptr != nullptr);
  selected_port = listen_and_serv_op_ptr->GetSelectedPort();
  std::string endpoint = paddle::string::Sprintf("127.0.0.1:%d", selected_port);
  attrs.insert({"endpoints", std::vector<std::string>({endpoint})});
  attrs.insert({"epmap", std::vector<std::string>({endpoint})});
  const f::VariableNameMap &inputs = {{"X", {"x1"}}};
  const f::VariableNameMap &outputs = {{"Out", {"Out"}}};

  auto send_op = f::OpRegistry::CreateOp("send", inputs, outputs, attrs);
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
  paddle::operators::ListenAndServOp::ResetPort();
}

TEST(SendRecvOp, CPUSparse) {
  std::atomic<bool> initialized;
  initialized = false;
  std::thread server_thread(StartServerNet, true, &initialized);
  while (!initialized) {
  }
  auto *listen_and_serv_op_ptr =
      static_cast<paddle::operators::ListenAndServOp *>(
          listen_and_serv_op.get());
  ASSERT_TRUE(listen_and_serv_op_ptr != nullptr);
  listen_and_serv_op_ptr->WaitServerReady();

  // local net
  f::Scope scope;
  p::CPUPlace place;
  p::CPUDeviceContext ctx(place);
  InitSelectedRowsInScope(place, &scope);
  scope.Var("RPC_CLIENT_VAR");
  f::AttributeMap attrs;
  selected_port = listen_and_serv_op_ptr->GetSelectedPort();
  std::string endpoint = paddle::string::Sprintf("127.0.0.1:%d", selected_port);
  attrs.insert({"endpoints", std::vector<std::string>({endpoint})});
  attrs.insert({"epmap", std::vector<std::string>({endpoint})});
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
  paddle::operators::ListenAndServOp::ResetPort();
}
