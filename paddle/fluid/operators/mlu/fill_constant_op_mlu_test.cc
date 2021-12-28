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

#include <gtest/gtest.h>

#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/operators/fill_constant_op.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(fill_constant);
USE_OP_DEVICE_KERNEL(fill_constant, MLU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type,float value,std::vector<int64_t> shape) {
  
  //output
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  //op maker
  f::AttributeMap attrs;
  attrs.insert({"value", value});
  attrs.insert({"shape", shape});
  f::VariableNameMap input;
  auto op =
      f::OpRegistry::CreateOp(op_type,
                              input,
                              {{"Out", {"Out"}}}, attrs);
  //op runner
  auto place = ctx.GetPlace();
  op->Run(*scope, place);

  ctx.Wait();

  std::vector<T> out_vec;
  
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  for (uint32_t i = 0; i < out_vec.size(); i++)
  {
    EXPECT_FLOAT_EQ(out_vec[i], value);
  }
 
}


TEST(fill_constant, MLU_fp32) {
  f::Scope scope;
  auto* ctx = p::DeviceContextPool::Instance().Get(p::MLUPlace(1));
  Compare<float>(&scope, *ctx,"fill_constant",1.0f,{6});
}
TEST(fill_constant2, MLU_fp32) {
  f::Scope scope;
  auto* ctx = p::DeviceContextPool::Instance().Get(p::MLUPlace(1));
  Compare<float>(&scope, *ctx,"fill_constant",2.0f,{6,3});
}
TEST(fill_constant3, MLU_fp32) {
  f::Scope scope;
  auto* ctx = p::DeviceContextPool::Instance().Get(p::MLUPlace(1));
  Compare<float>(&scope, *ctx,"fill_constant",3.0f,{2,3,4});
}
