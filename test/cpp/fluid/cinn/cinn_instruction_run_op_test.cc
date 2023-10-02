/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stdlib.h>

#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/kernel_registry.h"
#include "test/cpp/fluid/cinn/test_helper.h"

USE_OP_ITSELF(cinn_launch);
USE_OP_ITSELF(cinn_instruction_run);
USE_OP_ITSELF(elementwise_add);

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

class TestCinnInstructionRunOp : public ::testing::Test {
 public:
  const char* test_op_out_name = "test_op_out";
  const char* add_op_out_name = "add_op_out";
  std::unique_ptr<framework::OperatorBase> cinn_launch_op;
  std::unique_ptr<framework::OperatorBase> cinn_instruction_run_op;
  std::unique_ptr<framework::OperatorBase> elementwise_add_op;

  void SetUp() override {
    auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
        CreateOnlyElementwiseAddGraph("x", "y", test_op_out_name));

    // create necessary ops
    cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
        "cinn_launch",
        {{"X", {"x", "y"}}},
        {{"Out", {test_op_out_name}}},
        {{"compilation_key", compilation_key}});
    cinn_instruction_run_op = paddle::framework::OpRegistry::CreateOp(
        "cinn_instruction_run",
        {{"X", {"x", "y"}}},
        {{"Out", {test_op_out_name}}},
        {{"cached_index", 0}, {"instruction_index", 0}});
    elementwise_add_op =
        paddle::framework::OpRegistry::CreateOp("elementwise_add",
                                                {{"X", {"x"}}, {"Y", {"y"}}},
                                                {{"Out", {add_op_out_name}}},
                                                {{}});
  }

  void Compile(const platform::Place& place) {
    // check case: a compiled object not cached before cinn_launch_op run,
    // so a cinn_instruction_run_op will throw an error
    framework::Scope scope;
    InitVariablesWithRandomValue<float>({"x", "y"}, {10, 20}, place, &scope);
    scope.Var(test_op_out_name)->GetMutable<phi::DenseTensor>();
    ASSERT_THROW(cinn_instruction_run_op->Run(scope, place),
                 paddle::platform::EnforceNotMet);

    // run cinn_launch_op firstly to launch the compilation
    // of the above graph and cache two compiled results
    // of both type float and int
    cinn_launch_op->Run(scope, place);
    scope.EraseVars({"x", "y", test_op_out_name});
    scope.Var(test_op_out_name)->GetMutable<phi::DenseTensor>();
    InitVariablesWithRandomValue<int>({"x", "y"}, {30, 40}, place, &scope);
    cinn_launch_op->Run(scope, place);
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

TEST_F(TestCinnInstructionRunOp, CPU) {
  platform::CPUPlace place;
  Compile(place);
  framework::Scope scope1;
  RunAndCheck(place, &scope1);
  // the second run on the same place is to check the cache logic
  framework::Scope scope2;
  RunAndCheck(place, &scope2);
}

#ifdef PADDLE_WITH_CUDA
TEST_F(TestCinnInstructionRunOp, GPU) {
  platform::CUDAPlace place;
  Compile(place);
  framework::Scope scope1;
  RunAndCheck(place, &scope1);
  framework::Scope scope2;
  RunAndCheck(place, &scope2);
}
#endif

}  // namespace paddle::operators
