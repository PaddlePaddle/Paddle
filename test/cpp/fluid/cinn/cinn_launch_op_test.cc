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

#include "gtest/gtest.h"
#include "paddle/common/ddim.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/fluid/cinn/test_helper.h"

USE_OP_ITSELF(cinn_launch);
USE_OP_ITSELF(cinn_instruction_run);
USE_OP_ITSELF(elementwise_add);
PHI_DECLARE_double(eager_delete_tensor_gb);
PHI_DECLARE_bool(enable_pe_launch_cinn);
PHI_DECLARE_bool(enable_interpretercore_launch_cinn);
PHI_DECLARE_bool(enable_cinn_auto_tune);

PD_DECLARE_KERNEL(cinn_launch, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cinn_instruction_run, CPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
#ifdef PADDLE_WITH_CUDA
PD_DECLARE_KERNEL(cinn_launch, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(cinn_instruction_run, GPU, ALL_LAYOUT);
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
#endif

namespace paddle::operators {

using framework::paddle2cinn::CinnCompiler;

class TestCinnLaunchOp : public ::testing::Test {
 public:
  const char* test_op_out_name = "test_op_out";
  const char* add_op_out_name = "add_op_out";
  std::unique_ptr<framework::OperatorBase> cinn_launch_op;
  std::unique_ptr<framework::OperatorBase> elementwise_add_op;

  void SetUp() override {
    paddle::framework::InitDevices();
    platform::SetNumThreads(1);
    // cache test graph into CinnCompiler
    auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
        CreateOnlyElementwiseAddGraph("x", "y", test_op_out_name));

    // create cinn_launch_op and elementwise_add op
    cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
        "cinn_launch",
        {{"X", {"x", "y"}}},
        {{"Out", {test_op_out_name}}},
        {{"compilation_key", compilation_key}});
    elementwise_add_op =
        paddle::framework::OpRegistry::CreateOp("elementwise_add",
                                                {{"X", {"x"}}, {"Y", {"y"}}},
                                                {{"Out", {add_op_out_name}}},
                                                {{}});
  }

  void RunAndCheck(const platform::Place& place, framework::Scope* scope) {
    // Run ops and check the computation results
    InitVariablesWithRandomValue<float>({"x", "y"}, {10, 20}, place, scope);
    scope->Var(test_op_out_name)->GetMutable<phi::DenseTensor>();
    scope->Var(add_op_out_name)->GetMutable<phi::DenseTensor>();
    elementwise_add_op->Run(*scope, place);
    cinn_launch_op->Run(*scope, place);
    CompareOpResult<float>(scope->GetVar(test_op_out_name),
                           scope->GetVar(add_op_out_name));
  }

  void TearDown() override { CinnCompiler::GetInstance()->Clear(); }
};

TEST_F(TestCinnLaunchOp, TestRunCPUInstructionByPE) {
  framework::Scope scope1;
  RunAndCheck(platform::CPUPlace(), &scope1);
  // the second run on the same place is to check the cache logic
  framework::Scope scope2;
  RunAndCheck(platform::CPUPlace(), &scope2);
}

#ifdef PADDLE_WITH_CUDA
TEST_F(TestCinnLaunchOp, TestRunGPUInstructionByPE) {
  framework::Scope scope1;
  RunAndCheck(platform::CUDAPlace(), &scope1);
  framework::Scope scope2;
  RunAndCheck(platform::CUDAPlace(), &scope2);
}
#endif

TEST_F(TestCinnLaunchOp, TestRunCPUInstructionByCinnProgram) {
  // set FLAGS_enable_pe_launch_cinn=false to switch to use
  // default scheduler of CINN to execute the compiled program
  FLAGS_enable_pe_launch_cinn = false;
  FLAGS_enable_interpretercore_launch_cinn = false;
  framework::Scope scope1;
  RunAndCheck(platform::CPUPlace(), &scope1);
  framework::Scope scope2;
  RunAndCheck(platform::CPUPlace(), &scope2);
}

#ifdef PADDLE_WITH_CUDA
TEST_F(TestCinnLaunchOp, TestRunGPUInstructionByCinnProgram) {
  // set FLAGS_enable_pe_launch_cinn=false to switch to use
  // default scheduler of CINN to execute the compiled program
  FLAGS_enable_pe_launch_cinn = false;
  FLAGS_enable_interpretercore_launch_cinn = false;
  framework::Scope scope1;
  RunAndCheck(platform::CUDAPlace(), &scope1);
  framework::Scope scope2;
  RunAndCheck(platform::CUDAPlace(), &scope2);
}
#endif

TEST_F(TestCinnLaunchOp, TestRunWithAutoTuneEnabled) {
  FLAGS_enable_cinn_auto_tune = true;

  // currently only check on cpu, will add a test for gpu after CINN ready
  framework::Scope scope1;
  RunAndCheck(platform::CPUPlace(), &scope1);
  framework::Scope scope2;
  RunAndCheck(platform::CPUPlace(), &scope2);
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
