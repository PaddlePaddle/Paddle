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
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void test_transpose1_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("transpose-X", {2, 3, 4, 5});
  validator.DeclOutputVar("transpose-Out", {4, 2, 5, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"transpose-X"});
  desc.SetOutput("Out", {"transpose-Out"});
  desc.SetAttr("axis", std::vector<int>({2, 0, 3, 1}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(3);
}

template <typename TargetT>
void test_transpose2_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("transpose-X", {3, 4, 5});
  validator.DeclOutputVar("transpose-Out", {3, 5, 4});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("transpose");
  desc.SetInput("X", {"transpose-X"});
  desc.SetOutput("Out", {"transpose-Out"});
  desc.SetAttr("axis", std::vector<int>({0, 2, 1}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(transpose1_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_transpose1_op<::anakin::saber::NV>(ctx, true);
}

TEST(transpose2_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_transpose2_op<::anakin::saber::NV>(ctx, true);
}
#endif
#ifdef ANAKIN_X86_PLACE
TEST(transpose1_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_transpose2_op<::anakin::saber::X86>(ctx, false);
}

TEST(transpose2_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_transpose2_op<::anakin::saber::X86>(ctx, false);
}
#endif
}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(transpose);
USE_ANAKIN_CONVERTER(transpose);
