// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/eager_deletion_op_handle.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimization_var_info.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/program_desc.h"

USE_OP(mul);
USE_OP(cinn_launch);
USE_OP(elementwise_add);
namespace paddle {
namespace framework {

using Name2VarInfoMap =
    std::unordered_map<std::string, std::shared_ptr<ir::MemOptVarInfo>>;

static ProgramDesc BuildProgramInsideCinnLaunchOp() {
  ProgramDesc program;
  auto* block = program.MutableBlock(0);
  block->Var("var1");
  block->Var("var2");
  block->Var("var3");
  block->Var("var4");
  block->Var("var5");

  auto add_op = std::unique_ptr<OpDesc>(
      new OpDesc("elementwise_add", {{"X", {"var1"}}, {"Y", {"var2"}}},
                 {{"Out", {"var3"}}}, {}));
  block->AppendAllocatedOp(std::move(add_op));
  auto mul_op = std::unique_ptr<OpDesc>(new OpDesc(
      "mul", {{"X", {"var3"}}, {"Y", {"var4"}}}, {{"Out", {"var5"}}}, {}));
  block->AppendAllocatedOp(std::move(mul_op));
  return program;
}

static ProgramDesc BuildProgramWithCinnLaunchOp(
    const std::string& compilation_key) {
  // create a cinn_launch op
  ProgramDesc program;
  auto* block = program.MutableBlock(0);
  block->Var("var1");
  block->Var("var2");
  block->Var("var4");
  block->Var("var5");

  auto cinn_launch_op = std::unique_ptr<OpDesc>(
      new OpDesc("cinn_launch", {{"X", {"var1", "var2", "var4"}}},
                 {{"Out", {"var5"}}}, {{"compilation_key", compilation_key}}));
  block->AppendAllocatedOp(std::move(cinn_launch_op));
  return program;
}

struct TestPassContext {
  explicit TestPassContext(const ProgramDesc& program) {
    graph = std::make_unique<ir::Graph>(program);
    details::BuildStrategy build_strategy;
    details::ExecutionStrategy exec_strategy;
    exec_strategy.use_device_ = paddle::platform::kCUDA;
    executor.reset(new ParallelExecutor(platform::CUDAPlace(0), &scope,
                                        exec_strategy, build_strategy,
                                        graph.get()));
  }

  Scope scope;
  std::unique_ptr<ir::Graph> graph;
  std::unique_ptr<ParallelExecutor> executor;
};

TEST(ShareMemInfoToSubGraphPassTest, test_main_graph_share_var_info) {
  // add a subgraph to CinnCompiler
  auto subgraph = std::make_unique<ir::Graph>(BuildProgramInsideCinnLaunchOp());
  subgraph->GetOrInit<Name2VarInfoMap>(
      paddle2cinn::kMemOptVarInfoFromMainGraph);
  auto compilation_key =
      paddle2cinn::CinnCompiler::GetInstance()->AddGraph(std::move(subgraph));

  // build test data and apply pass
  auto context = std::make_unique<TestPassContext>(
      BuildProgramWithCinnLaunchOp(compilation_key));

  // check result
  const auto& result_subgraph =
      paddle2cinn::CinnCompiler::GetInstance()->FindGraph(compilation_key);
  const auto& shared_var_infos = result_subgraph.Get<Name2VarInfoMap>(
      paddle2cinn::kMemOptVarInfoFromMainGraph);
  ASSERT_EQ(shared_var_infos.size(), 4);
  EXPECT_EQ(shared_var_infos.count("var1"), 1);
  EXPECT_EQ(shared_var_infos.count("var5"), 1);
  EXPECT_EQ(shared_var_infos.at("var1").use_count(), 2);
  EXPECT_EQ(shared_var_infos.at("var5").use_count(), 2);
}

TEST(ShareMemInfoToSubGraphPassTest, test_sub_graph_take_var_info) {
  // build test data and apply pass
  auto context =
      std::make_unique<TestPassContext>(BuildProgramInsideCinnLaunchOp());
  auto& var_infos_shared = context->graph->GetOrInit<Name2VarInfoMap>(
      paddle2cinn::kMemOptVarInfoFromMainGraph);
  var_infos_shared = {
      {"var1", std::make_shared<ir::MemOptVarInfo>("var1", 1)},
      {"var2", std::make_shared<ir::MemOptVarInfo>("var2", 2)},
  };

  ir::MemOptVarInfoMapList mem_opt_var_infos(1);
  auto& var_infos = mem_opt_var_infos.front();
  var_infos = {{"var1", std::make_shared<ir::MemOptVarInfo>("var1", 1)},
               {"var2", std::make_shared<ir::MemOptVarInfo>("var2", 1)},
               {"var3", std::make_shared<ir::MemOptVarInfo>("var3", 1)},
               {"var4", std::make_shared<ir::MemOptVarInfo>("var4", 1)},
               {"var5", std::make_shared<ir::MemOptVarInfo>("var5", 1)}};
  auto share_pass =
      ir::PassRegistry::Instance().Get("share_mem_opt_info_to_subgraph_pass");
  share_pass->SetNotOwned(ir::kMemOptVarInfoMapList, &mem_opt_var_infos);
  share_pass->Apply(context->graph.get());

  // check result
  ASSERT_NE(var_infos.at("var1")->ParentHolder(), nullptr);
  ASSERT_NE(var_infos.at("var2")->ParentHolder(), nullptr);
  ASSERT_EQ(var_infos.at("var3")->ParentHolder(), nullptr);
  ASSERT_EQ(var_infos.at("var4")->ParentHolder(), nullptr);
  ASSERT_EQ(var_infos.at("var5")->ParentHolder(), nullptr);
}

}  // namespace framework
}  // namespace paddle
