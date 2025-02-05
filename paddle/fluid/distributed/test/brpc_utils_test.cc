/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/ps/service/brpc_utils.h"

#include <string>

#include "gtest/gtest.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle::framework {
class Variable;
}  // namespace paddle::framework

namespace framework = paddle::framework;
namespace platform = paddle::platform;

void CreateVarsOnScope(framework::Scope* scope,
                       phi::Place* place,
                       const phi::DeviceContext& ctx) {
  // var 1
  framework::Variable* var1 = scope->Var("x1");
  auto* tensor1 = var1->GetMutable<phi::DenseTensor>();
  tensor1->Resize(common::make_ddim({512, 8, 4, 2}));
  phi::LegacyLoD lod1;
  lod1.push_back(phi::Vector<size_t>({1, 3, 8}));
  tensor1->set_lod(lod1);
  tensor1->mutable_data<float>(*place);
  phi::funcs::set_constant(ctx, tensor1, static_cast<float>(31.9));

  // var 2
  framework::Variable* var2 = scope->Var("x2");
  auto* tensor2 = var2->GetMutable<phi::DenseTensor>();
  tensor2->Resize(common::make_ddim({1000, 64}));
  phi::LegacyLoD lod2;
  lod2.push_back(phi::Vector<size_t>({1, 1}));
  tensor2->set_lod(lod2);
  tensor2->mutable_data<int>(*place);
  phi::funcs::set_constant(ctx, tensor2, static_cast<int>(100));

  // var 3
  framework::Variable* var3 = scope->Var("x3");
  auto* slr = var3->GetMutable<phi::SelectedRows>();
  slr->set_height(564);
  auto* tensor3 = slr->mutable_value();
  auto* rows = slr->mutable_rows();
  tensor3->Resize(common::make_ddim({564, 128}));
  tensor3->mutable_data<float>(*place);
  phi::funcs::set_constant(ctx, tensor3, static_cast<float>(32.7));
  for (int i = 0; i < 564; ++i) rows->push_back(i);
}

void RunMultiVarMsg(phi::Place place) {
  framework::Scope scope;
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();
  auto& ctx = *pool.Get(place);
  CreateVarsOnScope(&scope, &place, ctx);

  ::paddle::distributed::MultiVariableMessage multi_msg;
  std::string message_name("se_de_test");
  std::vector<std::string> send_var_name = {"x1", "x2", "x3"};
  std::vector<std::string> recv_var_name = {};
  LOG(INFO) << "begin SerializeToMultiVarMsg";

  butil::IOBuf io_buf;
  ::paddle::distributed::SerializeToMultiVarMsgAndIOBuf(message_name,
                                                        send_var_name,
                                                        recv_var_name,
                                                        ctx,
                                                        &scope,
                                                        &multi_msg,
                                                        &io_buf);
  EXPECT_GT(multi_msg.ByteSizeLong(), static_cast<size_t>(0));

  // deserialize
  framework::Scope scope_recv;
  LOG(INFO) << "begin DeserializeFromMultiVarMsg";
  ::paddle::distributed::DeserializeFromMultiVarMsgAndIOBuf(
      multi_msg, &io_buf, ctx, &scope_recv);

  // check var1
  framework::Variable* var1 = scope_recv.FindVar("x1");
  auto* tensor1 = var1->GetMutable<phi::DenseTensor>();
  EXPECT_EQ(tensor1->dims(), common::make_ddim({512, 8, 4, 2}));
  // EXPECT_EQ(tensor1->lod(), phi::Vector<size_t>({1, 3, 8}));
  auto* tensor_data1 = const_cast<float*>(tensor1->data<float>());
  int tensor_numel1 = 512 * 8 * 4 * 2;
  for (int i = 0; i < tensor_numel1; ++i)
    EXPECT_FLOAT_EQ(tensor_data1[i], 31.9);

  // check var2
  framework::Variable* var2 = scope_recv.FindVar("x2");
  auto* tensor2 = var2->GetMutable<phi::DenseTensor>();
  EXPECT_EQ(tensor2->dims(), common::make_ddim({1000, 64}));
  // EXPECT_EQ(tensor2->lod(), phi::Vector<size_t>({1, 1}));
  auto* tensor_data2 = const_cast<int*>(tensor2->data<int>());
  int tensor_numel2 = 1000 * 64;
  for (int i = 0; i < tensor_numel2; ++i) EXPECT_EQ(tensor_data2[i], 100);

  // check var3
  framework::Variable* var3 = scope_recv.FindVar("x3");
  auto* slr = var3->GetMutable<phi::SelectedRows>();
  EXPECT_EQ(slr->rows().size(), 564UL);
  for (int i = 0; i < 564; ++i) {
    EXPECT_EQ(slr->rows()[i], i);
  }

  auto* tensor3 = slr->mutable_value();
  EXPECT_EQ(tensor3->dims(), common::make_ddim({564, 128}));
  auto* tensor_data3 = const_cast<float*>(tensor3->data<float>());
  int tensor_numel3 = 564 * 128;
  for (int i = 0; i < tensor_numel3; ++i)
    EXPECT_FLOAT_EQ(tensor_data3[i], 32.7);
}

TEST(MultiVarMsgCPU, Run) {
  phi::CPUPlace place;
  RunMultiVarMsg(place);
}

// #ifdef PADDLE_WITH_CUDA
// TEST(MultiVarMsgGPU, Run) {
//   phi::GPUPlace place;
//   RunMultiVarMsg(place);
// }
// #endif
