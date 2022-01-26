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

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device/mlu/device_context.h"
#include "paddle/fluid/platform/place.h"

namespace fw = paddle::framework;
namespace plat = paddle::platform;
namespace math = paddle::operators::math;

USE_OP(relu);
USE_OP_DEVICE_KERNEL(relu, MLU);

// relu
template <typename T>
inline T relu(T x) {
  return x > 0 ? x : 0.;
}

template <typename T>
inline T relu_grad_dx(T x, T out, T dout) {
  return out > 0 ? dout : 0;
}

template <typename T>
void Compare(fw::Scope* scope, const plat::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<fw::LoDTensor>();

  const int num = 10;
  std::vector<T> init_x;
  for (int64_t i = 0; i < num * num; ++i) {
    init_x.push_back(static_cast<T>(i - 50));
  }
  paddle::framework::TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({num, num});

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<fw::LoDTensor>();

  fw::AttributeMap attrs;
  auto op = fw::OpRegistry::CreateOp(op_type, {{"X", {"X"}}},
                                     {{"Out", {"Out"}}}, attrs);
  op->Run(*scope, place);

  ctx.Wait();

  // eval time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }

  ctx.Wait();

  gettimeofday(&end, NULL);
  int micros =
      (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> out_vec;
  paddle::framework::TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_FLOAT_EQ(out_vec[i], relu<T>(init_x[i]));
  }
}

template <typename T>
void CompareGrad(fw::Scope* scope, const plat::DeviceContext& ctx,
                 std::string op_type) {
  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<fw::LoDTensor>();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<fw::LoDTensor>();

  const int num = 10;
  std::vector<T> init_dout;
  for (int64_t i = 0; i < num * num; ++i) {
    init_dout.push_back(static_cast<T>(1.0));
  }

  std::vector<T> init_out;
  for (int64_t i = 0; i < num * num; ++i) {
    init_out.push_back(static_cast<T>(i - 50));
  }

  paddle::framework::TensorFromVector(init_dout, ctx, tensor_dout);
  tensor_dout->Resize({num, num});
  paddle::framework::TensorFromVector(init_out, ctx, tensor_out);
  tensor_out->Resize({num, num});

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<fw::LoDTensor>();

  // run
  auto place = ctx.GetPlace();
  fw::AttributeMap attrs;
  auto op = fw::OpRegistry::CreateOp(op_type,
                                     {{"Out@GRAD", {"DOut"}}, {"Out", {"Out"}}},
                                     {{"X@GRAD", {"DX"}}}, attrs);
  op->Run(*scope, place);

  ctx.Wait();

  // eval time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }

  ctx.Wait();

  gettimeofday(&end, NULL);
  int micros =
      (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> dx_vec;
  paddle::framework::TensorToVector(*tensor_dx, ctx, &dx_vec);

  ctx.Wait();

  for (uint32_t i = 0; i < dx_vec.size(); i++) {
    EXPECT_FLOAT_EQ(dx_vec[i],
                    relu_grad_dx<T>(dx_vec[i], init_out[i], init_dout[i]));
  }
}

TEST(relu, MLU_fp32) {
  fw::Scope scope;
  auto* ctx = plat::DeviceContextPool::Instance().Get(plat::MLUPlace(0));
  Compare<float>(&scope, *ctx, "relu");
}

TEST(relu_grad, MLU_fp32) {
  fw::Scope scope;
  auto* ctx = plat::DeviceContextPool::Instance().Get(plat::MLUPlace(0));
  CompareGrad<float>(&scope, *ctx, "relu_grad");
}
