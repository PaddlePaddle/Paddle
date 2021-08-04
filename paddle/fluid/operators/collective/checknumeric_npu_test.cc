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
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/operators/collective/gen_hccl_id_op_helper.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_allreduce_sum);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);
DECLARE_string(selected_npus);

template <typename T>
bool ContainsNan(const p::NPUDeviceContext& dev_ctx, aclrtStream stream,
                 const paddle::framework::Tensor* in) {
  // auto& dev_ctx =
  //    exe_ctx.template device_context<paddle::platform::NPUDeviceContext>();
  using Tensor = paddle::framework::Tensor;
  Tensor out(in->type());

  Tensor mean(in->type());
  mean.Resize({1});
  mean.mutable_data<T>(dev_ctx.GetPlace());
  std::vector<int> axes;
  for (int i = 0; i < in->dims().size(); ++i) {
    axes.push_back(i);
  }

  std::vector<T> vec;
  try {
    const auto& runner_mean = paddle::operators::NpuOpRunner(
        "ReduceMeanD", {*in}, {mean}, {{"axes", axes}, {"keep_dims", false}});
    TensorToVector(mean, dev_ctx, &vec);
  } catch (...) {
    LOG(WARNING) << "ContainsNan catch exception";
  }

  LOG(WARNING) << "reducesumd result:" << static_cast<float>(vec[0]);
  if (std::isnan(static_cast<float>(vec[0]))) {
    LOG(WARNING) << "contains nan";
    return true;
  }

  if (std::isinf(static_cast<float>(vec[0]))) {
    LOG(WARNING) << "contains inf";
    return true;
  }

  LOG(WARNING) << "Contains end";
  return false;
}

template <typename T>
bool CheckNumerics(const p::NPUDeviceContext& dev_ctx, aclrtStream stream,
                   const paddle::framework::Tensor* in) {
  using Tensor = paddle::framework::Tensor;
  Tensor out(in->type());
  out.Resize(in->dims());
  out.mutable_data<T>(dev_ctx.GetPlace());

  bool found_inf_data = false;

  try {
    const auto& runner = paddle::operators::NpuOpRunner(
        "CheckNumerics", {*in}, {out},
        {{"message", std::string("check_numberics")}});
    runner.Run(stream);
    dev_ctx.Wait();
  } catch (paddle::platform::EnforceNotMet& exception) {
    LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
    found_inf_data = true;
  } catch (...) {
    LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
    found_inf_data = true;
  }
  return found_inf_data;
}

template <typename T>
void Check(T value, int size = 2 * 512 * 8192) {
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

  TensorFromVector(init, ctx, tensor_x);
  VLOG(0) << "begin check 1";
  paddle::operators::CheckNumerics<T>(ctx, ctx.stream(), tensor_x);
  VLOG(0) << "end check 1";
  // CheckNumerics<T>(ctx, ctx.stream(), tensor_x);
  // VLOG(0) << "end check 2";
  ContainsNan<T>(ctx, ctx.stream(), tensor_x);
  // VLOG(0) << "end check 3";
  // ctx.Wait();
}

TEST(check_numeric, NPU) {
  auto inf = std::numeric_limits<float>::infinity();
  auto fp16_inf = static_cast<p::float16>(inf);
  auto nan = NAN;
  auto fp16_nan = static_cast<p::float16>(nan);

  try {
    // Normal
    VLOG(0) << "start normal";
    Check<p::float16>(static_cast<p::float16>(65546));
    Check<float>(static_cast<float>(1.0));

    // Inf
    VLOG(0) << "start inf";
    Check<p::float16>(fp16_inf);
    Check<float>(inf);

    // Nan
    VLOG(0) << "start nan";
    Check<p::float16>(fp16_nan);
    Check<float>(nan);
  } catch (...) {
    VLOG(0) << "catch execption";
  }
}
