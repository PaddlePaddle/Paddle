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

void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = var->GetMutable<f::LoDTensor>();
  tensor_x->Resize({10, 10});

  auto y = scope->Var("Y");
  auto tensor_y = var->GetMutable<f::LoDTensor>();
  tensor_y->Resize({10, 10});

  std::vector<float> init;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init.push_back(1.0);
  }

  TensorFromVector(init, ctx, tensor_x);
  TensorFromVector(init, ctx, tensor_y);

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out_var->GetMutable<f::LoDTensor>();
  out_tensor->Resize({10, 10});
  out_tensor->mutable_data<float>(place);  // allocate

  // run
  f::AttributeMap attrs;
  auto elementwise_add_op =
      f::OpRegistry::CreateOp("elementwise_add", {{"X", {"X"}}, {"Y", {"Y"}}},
                              {{"Out", {"Out"}}}, attrs);

  dropout_op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 2.0);
  }
}

TEST(Dropout, GPUDense) {
  f::Scope scope;
  p::CUDADeviceContext ctx(p::NPUPlace(0));
  Compare(scope, ctx);
}
