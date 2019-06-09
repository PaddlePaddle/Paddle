/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/sum.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"
#include "paddle/fluid/operators/sum_op.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
static void test_sum_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("sum_x1", {1, 2, 1, 2});
  validator.DeclInputVar("sum_x2", {1, 2, 1, 2});
  validator.DeclOutputVar("sum_out", {1, 2, 1, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("sum");
  desc.SetInput("X", {"sum_x1", "sum_x2"});
  desc.SetOutput("Out", {"sum_out"});

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(sum_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_sum_op<::anakin::saber::NV>(ctx, true);
}
#endif

TEST(sum_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_sum_op<::anakin::saber::X86>(ctx, false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(sum);
USE_CPU_ANAKIN_CONVERTER(sum);
#ifdef PADDLE_WITH_CUDA
USE_ANAKIN_CONVERTER(sum);
#endif
