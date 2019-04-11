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
#include "paddle/fluid/inference/anakin/convert/concat.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
void test_concat_op(const platform::DeviceContext& context, bool use_gpu) {
  std::unordered_set<std::string> parameters({""});
  framework::Scope scope;
  AnakinConvertValidation<TargetT, ::anakin::Precision::FP32> validator(
      parameters, &scope, context, use_gpu);
  validator.DeclInputVar("concat_x1", {1, 2, 1, 1});
  validator.DeclInputVar("concat_x2", {1, 3, 1, 1});
  validator.DeclInputVar("concat_x3", {1, 1, 1, 1});
  validator.DeclOutputVar("concat_out", {1, 6, 1, 1});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("concat");
  desc.SetInput("X", {"concat_x1", "concat_x2", "concat_x3"});
  desc.SetOutput("Out", {"concat_out"});

  int axis = 1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(concat_op, gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_concat_op<::anakin::saber::NV>(ctx, true);
}
#endif

TEST(concat_op, cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_concat_op<::anakin::saber::X86>(ctx, false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle
USE_OP(concat);
USE_CPU_ANAKIN_CONVERTER(concat);

#ifdef PADDLE_WITH_CUDA
USE_ANAKIN_CONVERTER(concat);
#endif
