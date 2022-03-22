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

#include <algorithm>
#include <map>
#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "cinn/common/target.h"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/operators/cinn/cinn_launch_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/core/ddim.h"

DECLARE_string(allow_cinn_ops);
DECLARE_string(deny_cinn_ops);

namespace paddle {
namespace framework {
namespace paddle2cinn {
using ir::Graph;
using ::cinn::common::Target;

namespace {
template <typename T, typename Alloc = std::allocator<T>>
std::ostream& operator<<(std::ostream& os, const std::vector<T, Alloc>& vec) {
  os << "{ ";
  for (auto e : vec) {
    os << e << " ";
  }
  os << "}\n";
  return os;
}

// Get compilation_key values
std::vector<std::string> GetCompilationKeys(const Graph& graph) {
  std::vector<std::string> compilation_keys;
  for (auto& node : graph.Nodes()) {
    if (node->IsOp() && node->Name() == kCinnLaunchOp) {
      compilation_keys.emplace_back(BOOST_GET_CONST(
          std::string, node->Op()->GetAttr(operators::kCompilationKey)));
    }
  }
  return compilation_keys;
}

// Extract op types from a graph
std::unordered_set<std::string> ExtractOpTypes(const Graph& graph) {
  std::unordered_set<std::string> op_types;
  for (auto& node : graph.Nodes()) {
    if (node->IsOp()) {
      op_types.emplace(node->Name());
    }
  }
  return op_types;
}

// Get inputs info
std::unordered_map<std::string, std::vector<int64_t>> GetInputsInfo(
    const std::string& key, const Graph& graph) {
  std::unordered_set<std::string> inputs;
  for (auto& node : graph.Nodes()) {
    if (node->IsOp() && node->Name() == kCinnLaunchOp) {
      if (BOOST_GET_CONST(std::string,
                          node->Op()->GetAttr(operators::kCompilationKey)) !=
          key) {
        continue;
      }
      for (auto in_var_name : node->Op()->InputArgumentNames()) {
        VLOG(4) << "get an input name: " << in_var_name;
        inputs.emplace(in_var_name);
      }
    }
  }

  std::unordered_map<std::string, std::vector<int64_t>> inputs_info;
  for (auto& node : graph.Nodes()) {
    if (node->IsVar() && inputs.count(node->Name())) {
      VLOG(4) << node->Name() << " : " << node->Var()->GetShape();
      inputs_info.emplace(node->Name(), node->Var()->GetShape());
    }
  }
  return inputs_info;
}

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
  mul_out->SetLoDLevel(0);
  mul_out->SetDataType(proto::VarType::FP32);
  mul_out->SetShape({1000, 100});
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
  add_out->SetLoDLevel(0);
  add_out->SetDataType(proto::VarType::FP32);
  add_out->SetShape({1000, 100});
  add_op->SetOutput("Out", {add_out->Name()});

  // relu
  auto* relu_op = global_block->AppendOp();
  relu_op->SetType("relu");
  relu_op->SetInput("X", {add_out->Name()});

  auto* relu_out = global_block->Var("RELU_OUT");
  relu_out->SetType(proto::VarType::LOD_TENSOR);
  relu_out->SetLoDLevel(0);
  relu_out->SetDataType(proto::VarType::FP32);
  relu_out->SetShape({1000, 100});
  relu_op->SetOutput("Out", {relu_out->Name()});
  program.Flush();
  return std::make_unique<Graph>(program);
}

}  // namespace

TEST(CinnCompilerTest, FlagController) {
  // init
  auto* cinn_compiler = CinnCompiler::GetInstance();
  auto cinn_pass = ir::PassRegistry::Instance().Get("build_cinn_pass");
  // apply build_cinn_pass & FLAGS_allow_cinn_ops="add"
  {
    FLAGS_allow_cinn_ops = "add";
    auto graph = CreateGraph();
    cinn_compiler->Clear();
    cinn_pass->Apply(graph.get());
    auto compilation_keys = GetCompilationKeys(*graph);
    ASSERT_EQ(compilation_keys.size(), 0);
  }
  // apply build_cinn_pass & FLAGS_allow_cinn_ops="mul;relu"
  {
    FLAGS_allow_cinn_ops = "mul;relu";
    auto graph = CreateGraph();
    cinn_compiler->Clear();
    cinn_pass->Apply(graph.get());
    auto compilation_keys = GetCompilationKeys(*graph);
    ASSERT_EQ(compilation_keys.size(), 2);
  }
  // apply build_cinn_pass & FLAGS_allow_cinn_ops="" &
  // FLAGS_deny_cinn_ops="relu"
  {
    FLAGS_allow_cinn_ops = "";
    FLAGS_deny_cinn_ops = "elementwise_add;relu";
    auto graph = CreateGraph();
    cinn_compiler->Clear();
    cinn_pass->Apply(graph.get());
    auto compilation_keys = GetCompilationKeys(*graph);
    ASSERT_EQ(compilation_keys.size(), 1);
    const auto& compiling_graph = cinn_compiler->FindGraph(compilation_keys[0]);
    auto op_types = ExtractOpTypes(compiling_graph);
    ASSERT_EQ(op_types.size(), 3);
    ASSERT_EQ(op_types.count("feed"), 1);
    ASSERT_EQ(op_types.count("mul"), 1);
    ASSERT_EQ(op_types.count("fetch"), 1);
  }
  // recover flags
  FLAGS_allow_cinn_ops = "";
  FLAGS_deny_cinn_ops = "";
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
  auto compilation_keys = GetCompilationKeys(*graph);
  ASSERT_EQ(compilation_keys.size(), 1);

  const auto& compilation_key = compilation_keys[0];
  auto* cinn_compiler = CinnCompiler::GetInstance();
  VLOG(4) << "The graph to be compiled:\n"
          << cinn_compiler->VizGraph(compilation_key);
  const auto& compiling_graph = cinn_compiler->FindGraph(compilation_key);
  viz_graph("compiling_graph.dot", const_cast<Graph*>(&compiling_graph));

  EXPECT_THROW(cinn_compiler->FindGraph("no_existed"),
               paddle::platform::EnforceNotMet);

  auto inputs_info = GetInputsInfo(compilation_key, *graph);
  std::unordered_map<std::string, LoDTensor> create_inputs;
  for (const auto& pair : inputs_info) {
    auto& tensor = create_inputs[pair.first];
    tensor.Resize(phi::make_ddim(pair.second));
    tensor.mutable_data<float>(platform::CPUPlace());
  }
  std::map<std::string, const LoDTensor*> input_tensors;
  std::for_each(create_inputs.begin(), create_inputs.end(),
                [&input_tensors](const auto& val) {
                  input_tensors.emplace(val.first, &val.second);
                });

  auto compile_fn = [&](const Target& target) {
    const auto& compiled_obj =
        cinn_compiler->Compile(compiling_graph, input_tensors, target);
    ASSERT_NE(compiled_obj.compiler, nullptr);
    ASSERT_NE(compiled_obj.runtime_program, nullptr);
    ASSERT_NE(compiled_obj.scope, nullptr);
    ASSERT_FALSE(compiled_obj.paddle2cinn_varmap.empty());
    ASSERT_NE(compiled_obj.launch_context, nullptr);
    const auto& cached_obj =
        cinn_compiler->Compile(compilation_key, input_tensors, target);
    ASSERT_EQ(reinterpret_cast<std::uint64_t>(&compiled_obj),
              reinterpret_cast<std::uint64_t>(&cached_obj));
    ASSERT_EQ(cached_obj.cached_index + 1, cinn_compiler->real_compiled_num());
    const auto& ret_obj =
        cinn_compiler->GetCompiledObject(cached_obj.cached_index);
    ASSERT_EQ(reinterpret_cast<std::uint64_t>(&compiled_obj),
              reinterpret_cast<std::uint64_t>(&ret_obj));
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
USE_OP_ITSELF(mul);
USE_OP_ITSELF(relu);
USE_OP_ITSELF(elementwise_add);
