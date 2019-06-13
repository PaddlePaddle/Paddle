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
#include "paddle/fluid/inference/anakin/convert/dropout.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void test_dropout_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("x", {1, 1, 2, 2});
  validator.DeclOutputVar("out", {1, 1, 2, 2});
  validator.DeclOutputVar("mask", {1, 1, 2, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("dropout");
  desc.SetInput("X", {"x"});
  desc.SetOutput("Out", {"out"});
  desc.SetOutput("Mask", {"mask"});

  float dropout_prob = 0.5;
  desc.SetAttr("dropout_prob", dropout_prob);
  desc.SetAttr("is_test", true);

  validator.SetOp(*desc.Proto());
  std::unordered_set<std::string> neglected_output = {"mask"};
  validator.Execute(1, neglected_output);
}

#ifdef PADDLE_WITH_CUDA
TEST(dropout_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_dropout_op<::anakin::saber::NV>(ctx, true);
}
#endif
#ifdef ANAKIN_X86_PLACE
TEST(dropout_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_dropout_op<::anakin::saber::X86>(ctx, false);
}
#endif
}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(dropout);
USE_ANAKIN_CONVERTER(dropout);
