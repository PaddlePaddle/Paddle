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
#include "cinn/common/type.h"
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/net_builder.h"  // need to remove after
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ::cinn::common::Target;
using ::cinn::common::Float;
using ::cinn::hlir::framework::GraphCompiler;
using ::cinn::hlir::framework::BuildScope;
using ::cinn::frontend::ProgramPass;
using ::cinn::hlir::framework::ApplyPass;

CinnCompiler* CinnCompiler::GetInstance() {
  static CinnCompiler instance;
  return &instance;
}

std::string CinnCompiler::AddGraph(std::unique_ptr<Graph> graph) {
  std::string graph_key;
  ProgramDesc program;
  GraphToProgram(*graph, &program);
  program.Proto()->SerializeToString(&graph_key);
  if (!graphs_.count(graph_key)) {
    graphs_[graph_key] = std::move(graph);
  } else {
    LOG(WARNING)
        << "The graph being added is already in CinnCompiler. Its key is:\n"
        << graph_key;
  }
  return graph_key;
}

const Graph& CinnCompiler::FindGraph(const std::string& graph_key) const {
  PADDLE_ENFORCE_NE(
      graphs_.count(graph_key), 0,
      platform::errors::InvalidArgument("Can not find the target graph: %s",
                                        graph_key.c_str()));
  return *graphs_.at(graph_key);
}

const CinnCompiledObject& CinnCompiler::Compile(
    const Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target) {
  CinnCacheKey cur_key(graph, input_tensors, target.arch_str());
  if (!cache_.count(cur_key)) {
    real_compiled_num_++;
    cache_[cur_key] = CompileGraph(graph, input_tensors, target);
  }
  return *cache_[cur_key];
}

const CinnCompiledObject& CinnCompiler::Compile(
    const std::string& compilation_key,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target) {
  const auto& graph = FindGraph(compilation_key);
  return Compile(graph, input_tensors, target);
}

std::unique_ptr<CinnCompiledObject> CinnCompiler::CompileGraph(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target) const {
  CinnGraphSymbolization symbol{real_compiled_num_, graph, target,
                                input_tensors};
  auto frontend_program = symbol();
  ProgramPass::Apply(&frontend_program, target, {"Decomposer"});
  auto cinn_graph = std::make_shared<::cinn::hlir::framework::Graph>(
      frontend_program, target);
  VLOG(4) << "The " << real_compiled_num_ << "-th compilation ("
          << target.arch_str() << "), and its related graph:\n"
          << cinn_graph->Visualize();
  ApplyPass(cinn_graph.get(), "OpFusion");
  auto scope = BuildScope(target, cinn_graph);
  GraphCompiler graph_compiler(target, scope, cinn_graph);
  GraphCompiler::CompileOptions options;
  options.with_instantiate_variables = false;
  auto compiled_res = graph_compiler.Build(options);
  auto compiled_obj = std::make_unique<CinnCompiledObject>();
  *compiled_obj = {std::move(compiled_res.runtime_program), scope,
                   symbol.var_model_to_program_map()};
  return compiled_obj;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
