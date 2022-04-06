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

#include "paddle/fluid/operators/cinn/cinn_launch_op.h"
#include <stdlib.h>
#include <mutex>
#include <random>
#include <string>
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/cinn/test_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/ddim.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP(cinn_launch);
USE_OP(cinn_instruction_run);
USE_OP_ITSELF(elementwise_add);
DECLARE_double(eager_delete_tensor_gb);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
#ifdef PADDLE_WITH_CUDA
PD_DECLARE_KERNEL(add, GPU, ALL_LAYOUT);
#endif

namespace paddle::operators {

using framework::paddle2cinn::CinnCompiler;

TEST(CinnLaunchOpTest, TestWithElementwiseAdd) {
  paddle::framework::InitDevices();
  platform::SetNumThreads(1);
  // cache test graph into CinnCompiler
  const std::string& test_op_out_name = "cinn_launch_op_out";
  const std::string& add_op_out_name = "add_op_out";
  auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
      CreateOnlyElementwiseAddGraph("x", "y", test_op_out_name));

  // create cinn_launch_op and elementwise_add op
  auto cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_launch", {{"X", {"x", "y"}}}, {{"Out", {test_op_out_name}}},
      {{"compilation_key", compilation_key}});
  auto elementwise_add_op = paddle::framework::OpRegistry::CreateOp(
      "elementwise_add", {{"X", {"x"}}, {"Y", {"y"}}},
      {{"Out", {add_op_out_name}}}, {{}});

  // Run ops and check the computation results
  auto run_and_check_fn = [&](const platform::Place& place) {
    framework::Scope scope;
    InitVariablesWithRandomValue<float>({"x", "y"}, {10, 20}, place, &scope);
    scope.Var(test_op_out_name)->GetMutable<LoDTensor>();
    scope.Var(add_op_out_name)->GetMutable<LoDTensor>();
    cinn_launch_op->Run(scope, place);
    elementwise_add_op->Run(scope, place);
    CompareOpResult<float>(scope.GetVar(test_op_out_name),
                           scope.GetVar(add_op_out_name));
  };
  FLAGS_eager_delete_tensor_gb = -1;

  // CPU
  run_and_check_fn(platform::CPUPlace());
  run_and_check_fn(platform::CPUPlace());
#ifdef PADDLE_WITH_CUDA
  // GPU
  run_and_check_fn(platform::CUDAPlace());
  run_and_check_fn(platform::CUDAPlace());
#endif
}

namespace details {
// Testing helper function used on CinnLaunchOpKernel in the following:
// firstly build test data, then check both expected and illegal situations

TEST(CinnLaunchOpHelperTest, TestPlaceToCinnTarget) {
  ASSERT_EQ(PlaceToCinnTarget(platform::CPUPlace()),
            ::cinn::common::DefaultHostTarget());
  ASSERT_EQ(PlaceToCinnTarget(platform::CUDAPlace(0)),
            ::cinn::common::DefaultNVGPUTarget());
  ASSERT_THROW(PlaceToCinnTarget(platform::XPUPlace()),
               paddle::platform::EnforceNotMet);
}

}  // namespace details
}  // namespace paddle::operators
