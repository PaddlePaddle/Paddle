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

#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/frontend/decomposer/use_decomposer.h"
#include "cinn/frontend/pass/use_program_pass.h"
#include "cinn/frontend/program_pass.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "cinn/hlir/framework/pass.h"
#include "cinn/hlir/pass/use_pass.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
namespace paddle2cinn {

using ir::Graph;
using ir::Node;
using inference::analysis::Dot;
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

const CinnCompiledObject& CinnCompiler::Compile(
    const Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target) {
  VLOG(1) << "-- The graph to be compiled is:\n" << VizGraph(graph);
  CinnCacheKey cur_key(graph, input_tensors, target.arch_str());
  bool exist = false;
  {
    AutoRDLock r_guard{&rwlock_};
    exist = cache_.count(cur_key) != 0;
  }
  if (!exist) {
    std::int64_t compiled_num = real_compiled_num_.fetch_add(1);
    auto compiled_res =
        CompileGraph(graph, input_tensors, target, compiled_num);
    AutoWRLock w_guard{&rwlock_};
    if (!cache_.count(cur_key)) {
      cache_[cur_key] = std::move(compiled_res);
    }
  }
  AutoRDLock guard{&rwlock_};
  const auto& cached_boj = *cache_[cur_key];
  return cached_boj;
}

const CinnCompiledObject& CinnCompiler::Compile(
    const std::string& compilation_key,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target) {
  const auto& graph = FindGraph(compilation_key);
  return Compile(graph, input_tensors, target);
}

std::string CinnCompiler::AddGraph(std::unique_ptr<Graph> graph) {
  std::string graph_key;
  ProgramDesc program;
  GraphToProgram(*graph, &program);
  program.Proto()->SerializeToString(&graph_key);

  PADDLE_ENFORCE_EQ(
      graphs_.count(graph_key), 0,
      platform::errors::PreconditionNotMet(
          "The graph to be added is already in CinnCompiler, which is:\n",
          VizGraph(graph_key).c_str()));
  graphs_[graph_key] = std::move(graph);
  VLOG(4) << "-- Add a graph into CinnCompiler, which is:\n"
          << VizGraph(graph_key);
  return graph_key;
}

const Graph& CinnCompiler::FindGraph(const std::string& graph_key) const {
  PADDLE_ENFORCE_NE(
      graphs_.count(graph_key), 0,
      platform::errors::PreconditionNotMet(
          "Can not find the target graph, of which the key is:\n%s",
          ReadableKey(graph_key).c_str()));
  return *graphs_.at(graph_key);
}

std::string CinnCompiler::VizGraph(const std::string& graph_key) const {
  const Graph& graph = FindGraph(graph_key);
  return VizGraph(graph);
}

std::string CinnCompiler::VizGraph(const Graph& graph) const {
  Dot dot;
  std::unordered_map<const Node*, std::string> node2dot;
  int id = 0;
  // Create nodes
  for (const Node* n : graph.Nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    if (n->IsOp()) {
      dot.AddNode(
          node_id,
          {Dot::Attr("shape", "box"), Dot::Attr("style", "rounded,filled,bold"),
           Dot::Attr("color", "#303A3A"), Dot::Attr("fontcolor", "#ffffff")},
          n->Name(), true);
    } else if (n->IsVar()) {
      auto label = n->Name();
      if (n->Var() && n->Var()->GetType() == proto::VarType::LOD_TENSOR) {
        auto shape = n->Var()->GetShape();
        std::vector<std::string> shape_str(shape.size());
        std::transform(shape.begin(), shape.end(), shape_str.begin(),
                       [](const auto& val) { return std::to_string(val); });
        label += "\n" + string::join_strings(shape_str, ',');
      }
      dot.AddNode(
          node_id,
          {Dot::Attr("shape", "box"), Dot::Attr("style", "rounded,filled,bold"),
           Dot::Attr("color", n->Var()->IsParameter() ? "#148b97" : "#dddddd"),
           Dot::Attr("fontcolor",
                     n->Var()->IsParameter() ? "#ffffff" : "#000000")},
          label, true);
    }
    node2dot[n] = node_id;
  }
  // Create edges
  for (const Node* n : graph.Nodes()) {
    const auto& src_id = node2dot.at(n);
    for (auto* out : n->outputs) {
      const auto& dest_id = node2dot.at(out);
      dot.AddEdge(src_id, dest_id, {});
    }
  }
  return dot.Build();
}

std::string CinnCompiler::ReadableKey(
    const std::string& compilation_key) const {
  proto::ProgramDesc desc;
  desc.ParseFromString(compilation_key);
  return desc.DebugString();
}

void CinnCompiler::Clear() {
  {
    AutoWRLock guard{&rwlock_};
    graphs_.clear();
    cache_.clear();
  }
  real_compiled_num_.store(0);
}

std::unique_ptr<CinnCompiledObject> CinnCompiler::CompileGraph(
    const ir::Graph& graph,
    const std::map<std::string, const LoDTensor*>& input_tensors,
    const Target& target, std::int64_t compiled_num) const {
  CinnGraphSymbolization symbol{compiled_num, graph, target, input_tensors};
  auto frontend_program = symbol();
  ProgramPass::Apply(&frontend_program, target, {"Decomposer"});
  auto cinn_graph = std::make_shared<::cinn::hlir::framework::Graph>(
      frontend_program, target);
  VLOG(1) << "-- The " << compiled_num << "-th compilation ("
          << target.arch_str() << "), and its related graph:\n"
          << cinn_graph->Visualize();
  ApplyPass(cinn_graph.get(), "OpFusion");
  auto scope = BuildScope(target, cinn_graph);

  auto fetch_ids = symbol.GetFetchIds();
  VLOG(4) << "All fetch var ids in CINN: "
          << string::join_strings(fetch_ids, ',');

  auto graph_compiler =
      std::make_unique<GraphCompiler>(target, scope, cinn_graph);
  GraphCompiler::CompileOptions options;
  options.with_instantiate_variables = false;
  auto compiled_res = graph_compiler->Build(options, std::move(fetch_ids));
  auto compiled_obj = std::make_unique<CinnCompiledObject>();
  *compiled_obj = {std::move(graph_compiler),
                   std::move(compiled_res.runtime_program), scope,
                   symbol.var_model_to_program_map()};
  return compiled_obj;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
