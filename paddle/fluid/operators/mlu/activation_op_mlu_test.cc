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
#include "paddle/fluid/operators/activation_op.h"

#include <gtest/gtest.h>

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"

// template <typename T>
// void Compare(f::Scope* scope, const p::DeviceContext& ctx,
//              std::string op_type) {
//   // init
//   auto x = scope->Var("X");
//   auto tensor_x = x->GetMutable<f::LoDTensor>();

//   std::vector<T> init;
//   init.push_back(static_cast<T>(1.0));
//   init.push_back(static_cast<T>(2.0));
//   init.push_back(static_cast<T>(3.0));
//   init.push_back(static_cast<T>(4.0));

//   TensorFromVector(init, ctx, tensor_x);
//   tensor_x->Resize({4});

//   ctx.Wait();

//   auto place = ctx.GetPlace();
//   auto out = scope->Var("Out");
//   auto tensor_out = out->GetMutable<f::LoDTensor>();

//   auto op =
//       f::OpRegistry::CreateOp(op_type, {{"X", {"X"}}}, {{"Out", {"Out"}}}, {});

//   op->Run(*scope, place);

//   std::vector<T> out_vec;
//   TensorToVector(*tensor_out, ctx, &out_vec);

//   ctx.Wait();

//   EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)4);
//   EXPECT_EQ(out_vec[0], static_cast<T>(1.0));
//   EXPECT_EQ(out_vec[1], static_cast<T>(2.0));
//   EXPECT_EQ(out_vec[2], static_cast<T>(3.0));
//   EXPECT_EQ(out_vec[3], static_cast<T>(4.0));
// }

// f::AttributeMap attrs;
//   float dropout_prob = 0.5;
//   attrs.insert({"fix_seed", 1});
//   attrs.insert({"seed", 3});
//   attrs.insert({"dropout_prob", dropout_prob});
//   auto dropout_op = f::OpRegistry::CreateOp(
//       "dropout", {{"X", {"X"}}}, {{"Out", {"Out"}}, {"Mask", {"Mask"}}}, attrs);

USE_OP(relu);
USE_OP_DEVICE_KERNEL(relu, MLU);

TEST(ActivationOpTest, TestReluOp) {
  paddle::platform::MLUPlace mlu_place(0);
  auto ctx = paddle::platform::DeviceContextPool::Instance().Get(mlu_place);

  paddle::framework::AttributeMap attrs;
  auto op = paddle::framework::OpRegistry::CreateOp(
                "relu", {{"X", {"X"}}}, {{"Out", {"Out"}}}, attrs);

  // init
  paddle::framework::Scope scope;
  auto x = scope.Var("X");
  auto tensor_x = x->GetMutable<paddle::framework::LoDTensor>();

  std::vector<float> init;
  init.push_back(-1.0);
  init.push_back(2.0);
  init.push_back(-3.0);
  init.push_back(4.0);

  TensorFromVector(init, *ctx, tensor_x);
  tensor_x->Resize({4});

  // ctx.Wait();

  auto out = scope.Var("Out");
  auto tensor_out = out->GetMutable<paddle::framework::LoDTensor>();


  op->Run(scope, mlu_place);
  LOG(INFO) << tensor_out;
}
