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

#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include <memory>
#include <set>
#include <utility>
#include "cinn/auto_schedule/auto_tuner.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/instruction.h"
#include "cinn/hlir/framework/scope.h"
#include "cinn/hlir/framework/tensor.h"
#include "cinn/runtime/cinn_runtime.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/cinn/cinn_op_helper.h"
#include "paddle/phi/core/ddim.h"

USE_OP(cinn_instruction_run);
namespace paddle {
namespace operators::details {

using framework::OpDesc;
using framework::ProgramDesc;
using framework::LoDTensor;
using framework::ir::Graph;
using framework::ParallelExecutor;
using framework::paddle2cinn::Name2VarInfoMap;
using CinnShape = ::cinn::hlir::framework::Shape;
using CinnInstruction = ::cinn::hlir::framework::Instruction;
using CinnRuntimeProgram = ::cinn::hlir::framework::Program;

const Graph& InitDefaultSubgraph() {
  static std::once_flag initialized;
  static std::unique_ptr<Graph> graph;
  std::call_once(initialized, [&]() {
    ProgramDesc program;
    auto* block = program.MutableBlock(0);
    auto* var1 = block->Var("var1");
    var1->SetPersistable(true);
    block->Var("var2");
    block->Var("var3");
    block->Var("var4");
    auto* var5 = block->Var("var5");
    var5->SetIsParameter(true);
    auto add_op = std::unique_ptr<OpDesc>(
        new OpDesc("elementwise_add", {{"X", {"var1"}}, {"Y", {"var2"}}},
                   {{"Out", {"var3"}}}, {}));
    block->AppendAllocatedOp(std::move(add_op));
    auto mul_op = std::unique_ptr<OpDesc>(new OpDesc(
        "mul", {{"X", {"var1"}}, {"Y", {"var2"}}}, {{"Out", {"var4"}}}, {}));
    block->AppendAllocatedOp(std::move(mul_op));
    auto res_op = std::unique_ptr<OpDesc>(
        new OpDesc("elementwise_add", {{"X", {"var3"}}, {"Y", {"var4"}}},
                   {{"Out", {"var5"}}}, {}));
    block->AppendAllocatedOp(std::move(res_op));
    graph = std::make_unique<Graph>(program);

    graph->Set<std::vector<std::string>>(
        framework::paddle2cinn::kInputVars,
        new std::vector<std::string>({"var1", "var2"}));
    graph->Set<std::vector<std::string>>(
        framework::paddle2cinn::kInternalVars,
        new std::vector<std::string>({"var3", "var4"}));
    graph->Set<std::vector<std::string>>(
        framework::paddle2cinn::kOutputVars,
        new std::vector<std::string>({"var5"}));
    graph->GetOrInit<Name2VarInfoMap>(
        framework::paddle2cinn::kMemOptVarInfoFromMainGraph);
  });
  return *graph.get();
}

CinnCompiledObject* InitDefaultCompiledObject() {
  static std::once_flag initialized;
  static auto compiled_obj = std::make_unique<CinnCompiledObject>();
  std::call_once(initialized, [result = compiled_obj.get()]() {
    auto& scope = result->scope;
    scope = std::make_shared<CinnScope>();
    std::vector<std::string> cinn_vars(
        {"cinn_var1", "cinn_var2", "cinn_var3", "cinn_var4", "cinn_var5"});

    // initialize variable and set data type
    for (const auto& var_name : cinn_vars) {
      scope->Var<CinnTensor>(var_name);
      scope->GetTensor(var_name)->set_type(::cinn::common::F32());
    }

    scope->GetTensor("cinn_var1")->Resize(CinnShape({3, 4}));
    scope->GetTensor("cinn_var2")->Resize(CinnShape({6, 7, 8}));
    scope->GetTensor("cinn_var3")->Resize(CinnShape({10, 16}));
    scope->GetTensor("cinn_var4")->Resize(CinnShape({10, 16}));
    scope->GetTensor("cinn_var5")->Resize(CinnShape({10, 16}));

    // input variables: var1, var2; output: var5
    // internal variables: var3 and var4, here var3 is retained
    // in result map, so the name will be used neither cinn_var3
    auto& paddle2cinn_varmap = result->paddle2cinn_varmap;
    paddle2cinn_varmap = {{"var1", "cinn_var1"},
                          {"var2", "cinn_var2"},
                          {"var3", "cinn_var3"},
                          {"var5", "cinn_var5"}};

    auto& runtime_program = result->runtime_program;
    std::vector<std::unique_ptr<CinnInstruction>> instructions;
    instructions.emplace_back(new CinnInstruction(
        cinn::common::DefaultHostTarget(), scope.get(),
        {"cinn_var1", "cinn_var2"}, {"cinn_var3"}, "elementwise_add"));
    instructions.emplace_back(
        new CinnInstruction(cinn::common::DefaultHostTarget(), scope.get(),
                            {"cinn_var1", "cinn_var2"}, {"cinn_var4"}, "mul"));
    instructions.emplace_back(new CinnInstruction(
        cinn::common::DefaultHostTarget(), scope.get(),
        {"cinn_var3", "cinn_var4"}, {"cinn_var5"}, "elementwise_add"));
    runtime_program =
        std::make_unique<CinnRuntimeProgram>(scope, std::move(instructions));
    result->cached_index = 110;
  });

  return compiled_obj.get();
}

class CinnLaunchContextTest : public ::testing::Test {
 public:
  std::unique_ptr<CinnLaunchContext> launch_context;
  CinnCompiledObject* compiled_obj;

  void SetUp() override {
    compiled_obj = InitDefaultCompiledObject();
    launch_context = std::make_unique<CinnLaunchContext>(InitDefaultSubgraph(),
                                                         *compiled_obj);
  }
};

TEST_F(CinnLaunchContextTest, TestConstructResult) {
  ASSERT_EQ(launch_context->IsVariableUsed("var1"), true);
  ASSERT_EQ(launch_context->IsVariableUsed("var2"), true);
  ASSERT_EQ(launch_context->IsVariableUsed("var3"), true);
  ASSERT_EQ(launch_context->IsVariableUsed("var4"), false);
  ASSERT_EQ(launch_context->IsVariableUsed("var5"), true);

  // check result of ExtractInternalVarNames
  ASSERT_EQ(launch_context->GetInternalVarNames(),
            std::unordered_set<std::string>({"var3", "cinn_var4"}));

  // check completeness of arguments list, and also check
  // the two name maps of the paddle->cinn and the reverse one
  // through the IsVariableUsed interface
  auto&& arguments = launch_context->FinalizeArguments();
  ASSERT_EQ(arguments.size(), 5);
  auto check_argument_fn = [&arguments, this](const std::string& var_name,
                                              const std::string& arg_name) {
    ASSERT_EQ(launch_context->IsVariableUsed(var_name), true);
    ASSERT_NO_THROW(launch_context->GetCinnBufferOfVar(var_name));
    ASSERT_GT(arguments.count(arg_name), 0);
    EXPECT_EQ(launch_context->GetCinnBufferOfVar(var_name),
              static_cast<cinn_buffer_t*>(arguments.at(arg_name)));
    auto* buffer = launch_context->GetCinnBufferOfVar(var_name);
    auto&& scope = compiled_obj->scope;
    ASSERT_EQ(framework::DDim(buffer->dims, buffer->dimensions),
              phi::make_ddim(scope->GetTensor(arg_name)->shape().data()));
  };
  check_argument_fn("var1", "cinn_var1");
  check_argument_fn("var2", "cinn_var2");
  check_argument_fn("var3", "cinn_var3");
  check_argument_fn("cinn_var4", "cinn_var4");
  check_argument_fn("var5", "cinn_var5");
}

TEST_F(CinnLaunchContextTest, TestCheckTensorEquivalent) {
  platform::CPUPlace place;
  framework::Scope scope;
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  auto* tensor2 = scope.Var("var2")->GetMutable<LoDTensor>();

  // dimension not equivalent
  tensor1->mutable_data<float>(phi::make_ddim({3, 5}), place);
  ASSERT_THROW(launch_context->CheckTensorEquivalent("var1", *tensor1),
               paddle::platform::EnforceNotMet);
  // data type not equivalent
  tensor2->mutable_data<int>(phi::make_ddim({6, 7, 8}), place);
  ASSERT_THROW(launch_context->CheckTensorEquivalent("var2", *tensor2),
               paddle::platform::EnforceNotMet);
}

TEST_F(CinnLaunchContextTest, TestBuildCompiledProgram) {
  platform::CPUPlace place;
  framework::Scope scope;
  ParallelExecutor* pe = nullptr;
  ASSERT_NO_THROW((pe = launch_context->InitializePE(place, &scope)));

  // check details of program build by compiled instructions
  const ProgramDesc& program = pe->Graph().OriginProgram();
  ASSERT_EQ(program.Size(), 1);
  const auto& block = program.Block(0);
  // vars
  std::set<std::string> var_names = block.LocalVarNames();
  ASSERT_EQ(var_names.size(), 5);
  for (auto&& var_name : var_names) {
    auto* var = block.FindVar(var_name);
    ASSERT_NE(var, nullptr);
    auto* buffer = launch_context->GetCinnBufferOfVar(var_name);
    ASSERT_EQ(framework::DDim(buffer->dims, buffer->dimensions),
              phi::make_ddim(var->GetShape()));
  }
  ASSERT_TRUE(block.FindVar("var1")->Persistable());
  ASSERT_FALSE(block.FindVar("var5")->Persistable());
  ASSERT_TRUE(block.FindVar("var5")->IsParameter());
  ASSERT_FALSE(block.FindVar("var1")->IsParameter());
  // ops
  ASSERT_EQ(block.OpSize(), 3);
  auto* op1 = block.Op(0);
  ASSERT_EQ(op1->Type(), "cinn_instruction_run");
  ASSERT_EQ(op1->Input(kX), std::vector<std::string>({"var1", "var2"}));
  ASSERT_EQ(op1->Output(kOutputs), std::vector<std::string>({"var3"}));
  ASSERT_EQ(op1->GetAttrIfExists<int64_t>(kCachedIndex), 110);
  ASSERT_EQ(op1->GetAttrIfExists<int64_t>(kInstructionIndex), 0);
  auto* op3 = block.Op(2);
  ASSERT_EQ(op3->Type(), "cinn_instruction_run");
  ASSERT_EQ(op3->Input(kX), std::vector<std::string>({"var3", "cinn_var4"}));
  ASSERT_EQ(op3->Output(kOutputs), std::vector<std::string>({"var5"}));
  ASSERT_EQ(op3->GetAttrIfExists<int64_t>(kCachedIndex), 110);
  ASSERT_EQ(op3->GetAttrIfExists<int64_t>(kInstructionIndex), 2);
}

// DEPRECATED(CtfGo): following test of callback assignment
// will be deprecated after we switch to pe
TEST_F(CinnLaunchContextTest, TestCallbackAssignment) {
  platform::CPUPlace place;
  framework::Scope scope;
  launch_context->UpdateCapturedEnv(scope, place);

  // assign external variables
  auto* tensor1 = scope.Var("var1")->GetMutable<LoDTensor>();
  float* data1 = tensor1->mutable_data<float>(phi::make_ddim({3, 4}), place);
  data1[0] = 9.99f;
  data1[10] = 19.99f;
  // check argument is set correctly and alloc/free callbacks work well
  auto* cinn_buffer = launch_context->GetCinnBufferOfVar("var1");
  ASSERT_EQ(cinn_buffer->memory, nullptr);
  cinn_buffer->external_malloc->operator()(nullptr, cinn_buffer);
  ASSERT_NE(cinn_buffer->memory, nullptr);
  ASSERT_EQ(cinn_buffer->num_elements(), 12);
  auto* shadow_data = reinterpret_cast<float*>(cinn_buffer->memory);
  EXPECT_FLOAT_EQ(shadow_data[0], 9.99f);
  EXPECT_FLOAT_EQ(shadow_data[10], 19.99f);
}

}  // namespace operators::details
}  // namespace paddle
