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

USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, NPU);
USE_OP(elementwise_sub);
USE_OP_DEVICE_KERNEL(elementwise_sub, NPU);

void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto y = scope->Var("Y");
  auto tensor_y = y->GetMutable<f::LoDTensor>();

  std::vector<float> init_x;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(1.0);
  }

  std::vector<float> init_y;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(2.0);
  }

  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({10, 10});
  TensorFromVector(init_y, ctx, tensor_y);
  tensor_y->Resize({10, 10});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({10, 10});
  tensor_out->mutable_data<float>(place);  // allocate

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(op_type, {{"X", {"X"}}, {"Y", {"Y"}}},
                                    {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();
  float expected;
  if (op_type == "elementwise_add") {
    expected = 3.0;
  } else if (op_type == "elementwise_sub") {
    expected = -1.0;
  }
  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 2.0);
  }
}

TEST(elementwise_add, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare(&scope, ctx);
}

TEST(elementwise_sub, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare(&scope, ctx);
}
