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

#include <stdio.h>
#include <cmath>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device/npu/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;

USE_OP(c_allreduce_sum);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);
DECLARE_string(selected_npus);

template <typename T>
bool Check(T value, int size = 2 * 512 * 8192) {
  f::Scope scope;
  auto x = scope.Var("in");
  auto& ctx = *dynamic_cast<p::NPUDeviceContext*>(
      p::DeviceContextPool::Instance().Get(p::NPUPlace(0)));
  auto place = ctx.GetPlace();

  auto tensor_x = x->GetMutable<f::LoDTensor>();
  tensor_x->Resize({size});
  tensor_x->mutable_data<T>(place);  // allocate

  std::vector<T> init;
  for (int64_t i = 0; i < size; ++i) {
    init.push_back(static_cast<T>(value));
  }

  paddle::framework::TensorFromVector(init, ctx, tensor_x);
  bool result = paddle::operators::ContainsNan(ctx, ctx.stream(), tensor_x);
  return result;
}

TEST(check_numeric, NPU) {
  auto inf = std::numeric_limits<float>::infinity();
  auto fp16_inf = static_cast<p::float16>(inf);
  auto nan = NAN;
  auto fp16_nan = static_cast<p::float16>(nan);

  bool result = false;
  // Normal
  VLOG(0) << "start normal";
  result = Check<p::float16>(static_cast<p::float16>(65546));
  ASSERT_FALSE(result);
  Check<float>(static_cast<float>(1.0));
  ASSERT_FALSE(result);

  // Inf
  VLOG(0) << "start inf";
  result = Check<p::float16>(fp16_inf);
  ASSERT_FALSE(result);
  result = Check<float>(inf);
  ASSERT_FALSE(result);

  // Nan
  VLOG(0) << "start nan";
  result = Check<p::float16>(fp16_nan);
  ASSERT_TRUE(result);
  result = Check<float>(nan);
  ASSERT_TRUE(result);
}
