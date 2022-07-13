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
#include <mutex>
#include <string>
#include <unordered_map>

#include "cinn/auto_schedule/auto_tuner.h"
#include "cinn/auto_schedule/tuning.h"
#include "cinn/common/target.h"
#include "cinn/common/type.h"
#include "cinn/frontend/optimize.h"
#include "cinn/frontend/syntax.h"
#include "cinn/hlir/framework/graph.h"
#include "cinn/hlir/framework/graph_compiler.h"
#include "gflags/gflags.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/paddle2cinn/build_cinn_pass.h"
#include "paddle/fluid/framework/paddle2cinn/cinn_graph_symbolization.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/analysis/dot.h"
#include "paddle/fluid/operators/cinn/cinn_launch_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(enable_cinn_allocate_oneshot);
DECLARE_bool(enable_pe_launch_cinn);
DECLARE_bool(enable_cinn_auto_tune);
namespace paddle {
namespace framework {
namespace paddle2cinn {

using ::cinn::auto_schedule::AutoTuner;
using ::cinn::common::Target;
using ::cinn::frontend::Optimize;
using ::cinn::hlir::framework::BuildScope;
using ::cinn::hlir::framework::GraphCompiler;
using inference::analysis::Dot;
using ir::Graph;
using ir::Node;

CinnCompiler *CinnCompiler::GetInstance() {
  static CinnCompiler *instance = new CinnCompiler();
  return instance;
}

const CinnCompiledObject &CinnCompiler::Compile(
    const Graph &graph,
    const std::map<std::string, const LoDTensor *> &input_tensors,
    const Target &target,
    void *stream) {
  VLOG(4) << "-- The graph to be compiled is:\n" << VizGraph(graph);
  CinnCacheKeyByAddress cur_key_by_address(
      graph, input_tensors, target.arch_str());
  CinnCacheKeyByStructure cur_key_by_struct;

  if (!cache_by_address_.count(cur_key_by_address)) {
    // generate the structure cache key
    cur_key_by_struct.SetKey(graph, input_tensors, target.arch_str());
    if (!cache_by_struct_.count(cur_key_by_struct)) {
      std::int64_t compiled_num = real_compiled_num_.fetch_add(1);
      auto compiled_res =
          CompileGraph(graph, input_tensors, target, compiled_num, stream);
      std::unique_lock<std::mutex> guard(lock_);
      // double check cache_by_struct_
      if (!cache_by_struct_.count(cur_key_by_struct)) {
        cache_by_struct_[cur_key_by_struct] = compiled_num;
        index2cache_.emplace(compiled_num, std::move(compiled_res));
      }
      // double check cache_by_address_
      if (!cache_by_address_.count(cur_key_by_address)) {
        cache_by_address_[cur_key_by_address] =
            cache_by_struct_.at(cur_key_by_struct);
      }
    } else {
      std::unique_lock<std::mutex> guard(lock_);
      // double check cache_by_address_
      if (!cache_by_address_.count(cur_key_by_address)) {
        cache_by_address_[cur_key_by_address] =
            cache_by_struct_.at(cur_key_by_struct);
      }
    }
  }
  return *index2cache_.at(cache_by_address_.at(cur_key_by_address));
}

const CinnCompiledObject &CinnCompiler::Compile(
    int64_t compilation_key,
    const std::map<std::string, const LoDTensor *> &input_tensors,
    const Target &target,
    void *stream) {
  const auto &graph = FindGraph(compilation_key);
  return Compile(graph, input_tensors, target, stream);
}

const CinnCompiledObject &CinnCompiler::GetCompiledObject(
    int64_t cached_index) const {
  auto res = index2cache_.find(cached_index);
  PADDLE_ENFORCE_NE(res,
                    index2cache_.end(),
                    platform::errors::InvalidArgument(
                        "Index(%ld) not found in cache", cached_index));
  return *res->second;
}

int64_t CinnCompiler::AddGraph(std::unique_ptr<Graph> graph) {
  int64_t graph_key = std::hash<Graph *>()((&(*graph)));
  PADDLE_ENFORCE_EQ(
      graphs_.count(graph_key),
      0,
      platform::errors::PreconditionNotMet(
          "The graph to be added is already in CinnCompiler, which is:\n",
          VizGraph(graph_key).c_str()));
  graphs_[graph_key] = std::move(graph);
  VLOG(4) << "-- Add a graph into CinnCompiler, which is:\n"
          << VizGraph(graph_key);
  return graph_key;
}

const Graph &CinnCompiler::FindGraph(int64_t graph_key) const {
  auto it = graphs_.find(graph_key);
  PADDLE_ENFORCE_NE(
      it,
      graphs_.end(),
      platform::errors::PreconditionNotMet(
          "Can not find the target graph, of which the key is: %lld",
          graph_key));
  return *it->second;
}

std::string CinnCompiler::VizGraph(int64_t graph_key) const {
  const Graph &graph = FindGraph(graph_key);
  return VizGraph(graph);
}

std::string CinnCompiler::VizGraph(const Graph &graph) const {
  Dot dot;
  std::unordered_map<const Node *, std::string> node2dot;
  int id = 0;
  // Create nodes
  for (const Node *n : graph.Nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    if (n->IsOp()) {
      dot.AddNode(node_id,
                  {Dot::Attr("shape", "box"),
                   Dot::Attr("style", "rounded,filled,bold"),
                   Dot::Attr("color", "#303A3A"),
                   Dot::Attr("fontcolor", "#ffffff")},
                  n->Name(),
                  true);
    } else if (n->IsVar()) {
      auto label = n->Name();
      if (n->Var() && n->Var()->GetType() == proto::VarType::LOD_TENSOR) {
        auto shape = n->Var()->GetShape();
        std::vector<std::string> shape_str(shape.size());
        std::transform(
            shape.begin(), shape.end(), shape_str.begin(), [](const auto &val) {
              return std::to_string(val);
            });
        label += "\n" + string::join_strings(shape_str, ',');
      }
      dot.AddNode(
          node_id,
          {Dot::Attr("shape", "box"),
           Dot::Attr("style", "rounded,filled,bold"),
           Dot::Attr("color", n->Var()->IsParameter() ? "#148b97" : "#dddddd"),
           Dot::Attr("fontcolor",
                     n->Var()->IsParameter() ? "#ffffff" : "#000000")},
          label,
          true);
    }
    node2dot[n] = node_id;
  }
  // Create edges
  for (const Node *n : graph.Nodes()) {
    const auto &src_id = node2dot.at(n);
    for (auto *out : n->outputs) {
      const auto &dest_id = node2dot.at(out);
      dot.AddEdge(src_id, dest_id, {});
    }
  }
  return dot.Build();
}

std::string CinnCompiler::SerializeKey(int64_t compilation_key) const {
  const auto &graph = FindGraph(compilation_key);

  ProgramDesc program;
  GraphToProgram(graph, &program);

  std::string serial_graph;
  program.Proto()->SerializeToString(&serial_graph);
  return serial_graph;
}

std::string CinnCompiler::ReadableKey(int64_t compilation_key) const {
  const auto &graph = FindGraph(compilation_key);

  ProgramDesc program;
  GraphToProgram(graph, &program);

  return program.Proto()->DebugString();
}

void CinnCompiler::Clear() {
  {
    std::unique_lock<std::mutex> guard(lock_);
    graphs_.clear();
    cache_by_address_.clear();
    cache_by_struct_.clear();
    index2cache_.clear();
  }
  real_compiled_num_.store(0);
}

void CinnCompiler::CheckCompiledValid(
    const ir::Graph &graph,
    const std::map<std::string, const LoDTensor *> &input_tensors,
    const CinnCompiledObject &compiled_obj) const {
  const auto &input_var_names = graph.Get<std::vector<std::string>>(kInputVars);
  const auto &output_var_names =
      graph.Get<std::vector<std::string>>(kOutputVars);
  auto *launch_context = compiled_obj.launch_context.get();
  // 1. check all of the output variables will be assigned by compiled program
  for (auto &&var_name : output_var_names) {
    PADDLE_ENFORCE_EQ(launch_context->IsVariableUsed(var_name),
                      true,
                      platform::errors::PreconditionNotMet(
                          "Variable(%s) not applied in CINN", var_name));
  }
  // 2. check all of the used input variables were correctly deduced by CINN.
  for (const auto &var_name : input_var_names) {
    // some input variables were not used by CINN because they were eliminated
    // by its optimized passes or some operators of it need less inputs
    if (!launch_context->IsVariableUsed(var_name)) {
      VLOG(4) << "Input variable" << var_name << " not used by cinn";
      continue;
    }
    launch_context->CheckTensorEquivalent(var_name,
                                          *input_tensors.at(var_name));
  }
}

std::unique_ptr<CinnCompiledObject> CinnCompiler::CompileGraph(
    const ir::Graph &graph,
    const std::map<std::string, const LoDTensor *> &input_tensors,
    const Target &target,
    std::int64_t compiled_num,
    void *stream) const {
  CinnGraphSymbolization symbol{compiled_num, graph, target, input_tensors};
  auto frontend_program = symbol();
  auto fetch_ids = symbol.GetFetchIds();
  VLOG(4) << "All fetch var ids in CINN: "
          << string::join_strings(fetch_ids, ',');

  auto cinn_graph = Optimize(&frontend_program, fetch_ids, target);
  VLOG(4) << "-- The " << compiled_num << "-th compilation ("
          << target.arch_str() << "), and its related graph:\n"
          << cinn_graph->Visualize();

  auto scope = BuildScope(target, cinn_graph);
  auto graph_compiler =
      std::make_unique<GraphCompiler>(target, scope, cinn_graph);
  GraphCompiler::CompileOptions options;
  options.with_instantiate_variables = false;
  if (!FLAGS_enable_pe_launch_cinn && !FLAGS_enable_cinn_allocate_oneshot) {
    options.with_buffer_handle_instruction_inserted = true;
  }
  std::unique_ptr<AutoTuner> auto_tuner;
  if (FLAGS_enable_cinn_auto_tune) {
    VLOG(4) << "Compile with auto-tune";
    auto_tuner = std::make_unique<AutoTuner>(target, cinn_graph.get());
    auto_tuner->Initialize(AutoTuner::Config(), graph_compiler.get());
    ::cinn::auto_schedule::TuningOptions tuning_options;
    tuning_options.num_measure_trials = 0;
    auto tuning_result = auto_tuner->Tune(tuning_options);
    options.Apply(tuning_result);
  }
  auto compiled_res =
      graph_compiler->Build(options, std::move(fetch_ids), stream);
  auto compiled_obj = std::make_unique<CinnCompiledObject>();
  *compiled_obj = {std::move(graph_compiler),
                   std::move(auto_tuner),
                   std::move(compiled_res.runtime_program),
                   scope,
                   symbol.var_model_to_program_map()};
  compiled_obj->cached_index = compiled_num;
  compiled_obj->launch_context =
      std::make_unique<operators::details::CinnLaunchContext>(graph,
                                                              *compiled_obj);
  CheckCompiledValid(graph, input_tensors, *compiled_obj);
  return compiled_obj;
}

}  // namespace paddle2cinn
}  // namespace framework
}  // namespace paddle
