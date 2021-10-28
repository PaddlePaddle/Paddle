// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/paddle2cinn/cinn_compiler.h"

#include <map>
#include <memory>
#include <string>

#include "cinn/common/target.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ::cinn::common::Target;

//  X -
//     | -> mul -> MUL_OUT -
//  Y -                     | -> elementwise_add -> ADD_OUT -> relu -> RELU_OUT
//                       Z -
std::unique_ptr<Graph> CreateGraph() {
  ProgramDesc program;
  auto* global_block = program.MutableBlock(0);
  // mul
  auto* x = global_block->Var("X");
  x->SetType(proto::VarType::LOD_TENSOR);
  x->SetLoDLevel(0);
  x->SetDataType(proto::VarType::FP32);
  x->SetShape({1000, 784});

  auto* y = global_block->Var("Y");
  y->SetType(proto::VarType::LOD_TENSOR);
  y->SetLoDLevel(0);
  y->SetDataType(proto::VarType::FP32);
  y->SetShape({784, 100});
  y->SetPersistable(true);
  y->SetIsParameter(true);

  auto* mul_op = global_block->AppendOp();
  mul_op->SetType("mul");
  mul_op->SetInput("X", {x->Name()});
  mul_op->SetInput("Y", {y->Name()});

  auto* mul_out = global_block->Var("MUL_OUT");
  mul_out->SetType(proto::VarType::LOD_TENSOR);
  mul_op->SetOutput("Out", {mul_out->Name()});

  // add
  auto* z = global_block->Var("Z");
  z->SetType(proto::VarType::LOD_TENSOR);
  z->SetLoDLevel(0);
  z->SetDataType(proto::VarType::FP32);
  z->SetShape({100});
  z->SetPersistable(true);
  z->SetIsParameter(true);

  auto* add_op = global_block->AppendOp();
  add_op->SetType("elementwise_add");
  add_op->SetInput("X", {mul_out->Name()});
  add_op->SetInput("Y", {z->Name()});

  auto* add_out = global_block->Var("ADD_OUT");
  add_out->SetType(proto::VarType::LOD_TENSOR);
  add_op->SetOutput("Out", {add_out->Name()});

  // relu
  auto* relu_op = global_block->AppendOp();
  relu_op->SetType("relu");
  relu_op->SetInput("X", {add_out->Name()});

  auto* relu_out = global_block->Var("RELU_OUT");
  relu_out->SetType(proto::VarType::LOD_TENSOR);
  relu_op->SetOutput("Out", {relu_out->Name()});
  program.Flush();
  return std::make_unique<Graph>(program);
}

TEST(CinnCompilerTest, Compile) {
  auto viz_pass = ir::PassRegistry::Instance().Get("graph_viz_pass");
  auto cinn_pass = ir::PassRegistry::Instance().Get("build_cinn_pass");
  auto viz_graph = [&viz_pass](const std::string& viz_path, Graph* graph) {
    viz_pass->Erase("graph_viz_path");
    viz_pass->Set("graph_viz_path", new std::string(viz_path));
    viz_pass->Apply(graph);
  };

  // create a graph
  auto graph = CreateGraph();
  viz_graph("origin_graph.dot", graph.get());
  // apply build_cinn_pass
  cinn_pass->Apply(graph.get());
  viz_graph("processed_graph.dot", graph.get());
  // get the compilation_key
  std::vector<std::string> compilation_keys;
  for (auto& node : graph->Nodes()) {
    if (node->IsOp() && node->Name() == kCinnLaunchOp) {
      compilation_keys.emplace_back(
          BOOST_GET_CONST(std::string, node->Op()->GetAttr(kCompilationKey)));
    }
  }
  ASSERT_EQ(compilation_keys.size(), 1);

  const auto& compilation_key = compilation_keys[0];
  auto* cinn_compiler = CinnCompiler::GetInstance();
  const auto& compiling_graph = cinn_compiler->FindGraph(compilation_key);
  // viz_graph("compiling_graph.dot", const_cast<Graph*>(&compiling_graph));

  EXPECT_THROW(cinn_compiler->FindGraph("no_existed"),
               paddle::platform::EnforceNotMet);

  LoDTensor tensor1, tensor2, tensor3;
  tensor1.Resize({1000, 784});
  tensor2.Resize({784, 100});
  tensor3.Resize({100});
  tensor1.mutable_data<float>(platform::CPUPlace());
  tensor2.mutable_data<float>(platform::CPUPlace());
  tensor3.mutable_data<float>(platform::CPUPlace());
  std::map<std::string, const LoDTensor*> input_tensors = {
      {"X", &tensor1}, {"Y", &tensor2}, {"Z", &tensor3}};

  auto compile_fn = [&](const Target& target) {
    const auto& compiled_obj =
        cinn_compiler->Compile(compiling_graph, input_tensors, target);
    ASSERT_NE(compiled_obj.runtime_program, nullptr);
    ASSERT_NE(compiled_obj.scope, nullptr);
    ASSERT_FALSE(compiled_obj.paddle2cinn_varmap.empty());
    const auto& cached_obj =
        cinn_compiler->Compile(compilation_key, input_tensors, target);
    ASSERT_EQ(reinterpret_cast<std::uint64_t>(&compiled_obj),
              reinterpret_cast<std::uint64_t>(&cached_obj));
  };

  // GPU Compilation
  compile_fn(::cinn::common::DefaultNVGPUTarget());
  ASSERT_EQ(cinn_compiler->real_compiled_num(), 1);
  // CPU Compilation
  compile_fn(::cinn::common::DefaultHostTarget());
  ASSERT_EQ(cinn_compiler->real_compiled_num(), 2);
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle

USE_PASS(build_cinn_pass);
USE_PASS(graph_viz_pass);
