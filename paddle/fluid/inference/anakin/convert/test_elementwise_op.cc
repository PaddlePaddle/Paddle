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
#include "paddle/fluid/inference/anakin/convert/elementwise.h"
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

template <typename TargetT>
static void test_elementwise_op(const std::string& op_type,
                                const platform::DeviceContext& context,
                                bool use_gpu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  AnakinConvertValidation<TargetT> validator(parameters, &scope, context,
                                             use_gpu);
  validator.DeclInputVar("x", {1, 1, 2, 2});
  validator.DeclInputVar("y", {1, 1, 2, 2});
  validator.DeclOutputVar("out", {1, 1, 2, 2});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType(op_type);
  desc.SetInput("X", {"x"});
  desc.SetInput("Y", {"y"});
  desc.SetOutput("Out", {"out"});

  int axis = -1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());
  validator.Execute(1);
}

#ifdef PADDLE_WITH_CUDA
TEST(elementwise_op, native_add_gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_elementwise_op<::anakin::saber::NV>("elementwise_add", ctx, true);
}
TEST(elementwise_op, native_mul_gpu) {
  platform::CUDAPlace gpu_place(0);
  platform::CUDADeviceContext ctx(gpu_place);
  test_elementwise_op<::anakin::saber::NV>("elementwise_mul", ctx, true);
}
#endif

TEST(elementwise_op, native_add_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_elementwise_op<::anakin::saber::X86>("elementwise_add", ctx, false);
}

TEST(elementwise_op, native_mul_cpu) {
  platform::CPUPlace cpu_place;
  platform::CPUDeviceContext ctx(cpu_place);
  test_elementwise_op<::anakin::saber::X86>("elementwise_mul", ctx, false);
}

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(elementwise_add);
USE_OP(elementwise_mul);
#ifdef PADDLE_WITH_CUDA
USE_ANAKIN_CONVERTER(elementwise_add);
USE_ANAKIN_CONVERTER(elementwise_mul);
#endif

USE_CPU_ANAKIN_CONVERTER(elementwise_add);
USE_CPU_ANAKIN_CONVERTER(elementwise_mul);
