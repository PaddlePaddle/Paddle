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
void test_reshape1_op(const platform::DeviceContext& context, bool use_gpu) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);

  // validator.DeclInputVar("reshape-X", {2, 3, 3, 1});
  // validator.DeclOutputVar("reshape-Out", {3, 2, 1, 3});
  validator.DeclInputVar("reshape-X", {1, 2, 4, 1});
  validator.DeclOutputVar("reshape-Out", {1, 8, 1, 1});

  framework::OpDesc desc;
  desc.SetType("reshape");
  desc.SetInput("X", {"reshape-X"});
  desc.SetOutput("Out", {"reshape-Out"});
  // desc.SetAttr("shape", std::vector<int>({3, 2, 1, 3}));
  desc.SetAttr("shape", std::vector<int>({1, 8, 1, 1}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";
  validator.Execute(1);
}

template <typename TargetT>
void test_reshape2_op(const platform::DeviceContext& context, bool use_gpu) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);

  validator.DeclInputVar("reshape-X", {1, 2, 4});
  validator.DeclOutputVar("reshape-Out", {1, 4, 2});

  framework::OpDesc desc;
  desc.SetType("reshape");
  desc.SetInput("X", {"reshape-X"});
  desc.SetOutput("Out", {"reshape-Out"});
  // desc.SetAttr("shape", std::vector<int>({3, 2, 1, 3}));
  desc.SetAttr("shape", std::vector<int>({0, -1, 2}));

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";
  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(reshape1_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_reshape1_op<::anakin::saber::NV>(ctx, true);
}

TEST(reshape2_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_reshape2_op<::anakin::saber::NV>(ctx, true);
}
#endif

TEST(reshape1_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_reshape2_op<::anakin::saber::X86>(ctx, false);
}

TEST(reshape2_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_reshape2_op<::anakin::saber::X86>(ctx, false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(reshape);
USE_CPU_ANAKIN_CONVERTER(reshape);

#ifdef PADDLE_WITH_CUDA
USE_ANAKIN_CONVERTER(reshape);
#endif
