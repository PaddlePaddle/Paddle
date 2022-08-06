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
#include "paddle/fluid/operators/cinn/test_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/phi/core/kernel_registry.h"

USE_OP(cinn_launch);
USE_OP(cinn_instruction_run);
USE_OP_ITSELF(elementwise_add);

PD_DECLARE_KERNEL(add, CPU, ALL_LAYOUT);
#ifdef PADDLE_WITH_CUDA
PD_DECLARE_KERNEL(add, KPS, ALL_LAYOUT);
#endif

namespace paddle::operators {

using framework::paddle2cinn::CinnCompiler;

TEST(CinnInstructionOpTest, TestWithElementwiseAdd) {
  paddle::framework::InitDevices();
  platform::SetNumThreads(1);
  // cache test graph into CinnCompiler
  const std::string& test_op_out_name = "cinn_instruction_run_op_out";
  const std::string& add_op_out_name = "add_op_out";
  auto compilation_key = CinnCompiler::GetInstance()->AddGraph(
      CreateOnlyElementwiseAddGraph("x", "y", test_op_out_name));

  // create necessary ops
  auto cinn_instruction_run_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_instruction_run",
      {{"X", {"x", "y"}}},
      {{"Out", {test_op_out_name}}},
      {{"cached_index", 0}, {"instruction_index", 0}});

  auto cinn_launch_op = paddle::framework::OpRegistry::CreateOp(
      "cinn_launch",
      {{"X", {"x", "y"}}},
      {{"Out", {test_op_out_name}}},
      {{"compilation_key", compilation_key}});

  // check case: a compiled object not cached before cinn_launch_op run,
  // so a cinn_instruction_run_op will throw an error
  framework::Scope scope;
  platform::CPUPlace place;
  InitVariablesWithRandomValue<float>({"x", "y"}, {10, 20}, place, &scope);
  scope.Var(test_op_out_name)->GetMutable<LoDTensor>();
  ASSERT_THROW(cinn_instruction_run_op->Run(scope, place),
               paddle::platform::EnforceNotMet);
  // run cinn_launch_op firstly to launch the compilation
  // of the above graph and cache two compiled results
  // of both type float and int
  cinn_launch_op->Run(scope, place);
  scope.EraseVars({"x", "y", test_op_out_name});
  scope.Var(test_op_out_name)->GetMutable<LoDTensor>();
  InitVariablesWithRandomValue<int>({"x", "y"}, {30, 40}, place, &scope);
  cinn_launch_op->Run(scope, place);

  // Run ops and check the computation results
  auto run_and_check_fn = [&](const platform::Place& place) {
    framework::Scope scope;
    scope.Var(test_op_out_name)->GetMutable<LoDTensor>();
    scope.Var(add_op_out_name)->GetMutable<LoDTensor>();
    auto elementwise_add_op =
        paddle::framework::OpRegistry::CreateOp("elementwise_add",
                                                {{"X", {"x"}}, {"Y", {"y"}}},
                                                {{"Out", {add_op_out_name}}},
                                                {{}});

    // 1. check on type float
    InitVariablesWithRandomValue<float>({"x", "y"}, {10, 20}, place, &scope);
    cinn_instruction_run_op->SetAttr("cached_index", 0);
    cinn_instruction_run_op->Run(scope, place);
    elementwise_add_op->Run(scope, place);
    CompareOpResult<float>(scope.GetVar(test_op_out_name),
                           scope.GetVar(add_op_out_name));

    // 2. check on type int to indicate cinn_instruction_run op
    // can mutable data according compiled result
    scope.EraseVars({"x", "y", test_op_out_name, add_op_out_name});
    scope.Var(test_op_out_name)->GetMutable<LoDTensor>();
    scope.Var(add_op_out_name)->GetMutable<LoDTensor>();

    InitVariablesWithRandomValue<int>({"x", "y"}, {30, 40}, place, &scope);
    cinn_instruction_run_op->SetAttr("cached_index", 1);
    cinn_instruction_run_op->Run(scope, place);
    // need reconstruct elementwise_add_op to choose a new kernel with type int
    elementwise_add_op =
        paddle::framework::OpRegistry::CreateOp("elementwise_add",
                                                {{"X", {"x"}}, {"Y", {"y"}}},
                                                {{"Out", {add_op_out_name}}},
                                                {{}});
    elementwise_add_op->Run(scope, place);
    CompareOpResult<int>(scope.GetVar(test_op_out_name),
                         scope.GetVar(add_op_out_name));
  };

  // CPU
  run_and_check_fn(platform::CPUPlace());
  run_and_check_fn(platform::CPUPlace());
#ifdef PADDLE_WITH_CUDA
  // GPU
  run_and_check_fn(platform::CUDAPlace());
  run_and_check_fn(platform::CUDAPlace());
#endif
}

}  // namespace paddle::operators
