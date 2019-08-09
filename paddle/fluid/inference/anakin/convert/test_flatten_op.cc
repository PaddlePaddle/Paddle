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
void test_flatten_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("flatten-X", {3, 10, 10, 4});
  validator.DeclOutputVar("flatten-Out", {3, 400, 1, 1});
  framework::OpDesc desc;
  desc.SetType("flatten");
  desc.SetInput("X", {"flatten-X"});
  desc.SetOutput("Out", {"flatten-Out"});
  desc.SetAttr("axis", 1);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(5);
}

#ifdef PADDLE_WITH_CUDA
TEST(flatten_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_flatten_op<::anakin::saber::NV>(ctx, true);
}
#endif

TEST(flatten_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_flatten_op<::anakin::saber::X86>(ctx, false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(reshape);
USE_OP_ITSELF(flatten);
USE_CPU_ANAKIN_CONVERTER(flatten);
#ifdef PADDLE_WITH_CUDA
USE_ANAKIN_CONVERTER(flatten);
#endif
