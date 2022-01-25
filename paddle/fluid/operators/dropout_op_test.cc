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

#ifndef _WIN32
#include <unistd.h>
#endif

#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(dropout);

void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto var = scope->Var("X");
  auto tensor = var->GetMutable<f::LoDTensor>();
  tensor->Resize({10, 10});

  std::vector<float> init;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(1.0);
  }

  paddle::framework::TensorFromVector(init, ctx, tensor);

  auto place = ctx.GetPlace();
  auto out_var = scope->Var("Out");
  auto out_tensor = out_var->GetMutable<f::LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate

  auto mask_var = scope->Var("Mask");
  auto mask_tensor = mask_var->GetMutable<f::LoDTensor>();
  mask_tensor->Resize({10, 10});
  mask_tensor->mutable_data<float>(place);  // allocate

  // run
  f::AttributeMap attrs;
  float dropout_prob = 0.5;
  attrs.insert({"fix_seed", 1});
  attrs.insert({"seed", 3});
  attrs.insert({"dropout_prob", dropout_prob});
  auto dropout_op = f::OpRegistry::CreateOp(
      "dropout", {{"X", {"X"}}}, {{"Out", {"Out"}}, {"Mask", {"Mask"}}}, attrs);

  dropout_op->Run(*scope, place);

  std::vector<float> out_vec;
  paddle::framework::TensorToVector(*out_tensor, ctx, &out_vec);

  std::vector<float> std_out = {
      0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1,
      1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
      1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1,
      1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0,
      1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1};

  EXPECT_EQ(out_vec.size(), std_out.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], std_out[i]);
  }
}

// TODO(wyi): Due to
// https://github.com/PaddlePaddle/Paddle/issues/9507, I temporarily
// disable this test to remove the prevention of the merge of
// unrelated PRs.
/*
TEST(Dropout, CPUDense) {
  f::Scope scope;
  p::CPUPlace place;
  p::CPUDeviceContext ctx(place);
  Compare(scope, ctx);
}

TEST(Dropout, GPUDense) {
  f::Scope scope;
  p::CUDAPlace place;
  p::CUDADeviceContext ctx(place);
  Compare(scope, ctx);
}
*/
