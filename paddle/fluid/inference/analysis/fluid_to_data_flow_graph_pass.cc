/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <string>
#include <vector>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/analysis/fluid_to_data_flow_graph_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

bool FluidToDataFlowGraphPass::Initialize(Argument *argument) {
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument);
  if (argument->origin_program_desc) {
    LOG(WARNING) << "argument's origin_program_desc is already set, might "
                    "duplicate called";
  }
  if (!argument->fluid_model_program_path) {
    ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_dir);
    argument->fluid_model_program_path.reset(
        new std::string(*argument->fluid_model_dir + "/__model__"));
  }
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument->fluid_model_program_path);
  auto program = LoadProgramDesc(*argument->fluid_model_program_path);
  argument->origin_program_desc.reset(
      new framework::proto::ProgramDesc(program));

  if (!argument->main_dfg) {
    argument->main_dfg.reset(new DataFlowGraph);
  }
  desc_ = argument->origin_program_desc.get();
  return true;
}

bool FluidToDataFlowGraphPass::Finalize() { return true; }

void FluidToDataFlowGraphPass::Run(DataFlowGraph *graph) {
  PADDLE_ENFORCE(graph);
  PADDLE_ENFORCE(desc_);
  // insert vars
  // The `var2id` keeps a map from a variable's name to its Node-id, the Node-id
  // will keep updating to its latest alias during the graph-building.
  std::unordered_map<std::string, size_t> var2id;
  auto &main_block = desc_->blocks(framework::kRootBlockIndex);
  for (int i = 0; i < main_block.vars_size(); i++) {
    const auto &var = main_block.vars(i);
    auto *v = graph->nodes.Create(Node::Type::kValue);
    v->SetName(var.name());
    v->SetPbDesc(const_cast<void *>(static_cast<const void *>(&var)));
    v->SetPbMsg(var.SerializeAsString());
    var2id[var.name()] = v->id();
  }

  // The variables in a SSA can only write once, so if a variable is written
  // multiple times(quite common in our ProgramDesc design), multiple alias
  // Nodes of this variable will be created, and each will just write once.

  // An set that keep all the names of the variables(the original, not alias)
  // that have been written(as outputs). Once an Op's output variable hit the
  // set, it should create a new alias and update the global alias for this
  // variable. And that make a Data Flow Graph a SSA.
  std::unordered_set<Node *> unique_written_vars;
  for (int i = 0; i < main_block.ops_size(); i++) {
    const auto &op = main_block.ops(i);
    auto *o = graph->nodes.Create(Node::Type::kFunction);
    o->SetName(op.type());
    static_cast<Function *>(o)->SetFuncType(op.type());
    // Link to the original protobuf message's memory, make it easier to
    // generate from a data flow graph to fluid ProgramDesc.
    o->SetPbDesc(const_cast<void *>(static_cast<const void *>(&op)));
    o->SetPbMsg(op.SerializeAsString());

    // set inputs and outputs
    for (int j = 0; j < op.inputs_size(); j++) {
      auto &in_var = op.inputs(j);
      for (int k = 0; k < in_var.arguments_size(); k++) {
        auto *in = graph->nodes.GetMutable(var2id.at(in_var.arguments(k)));
        in->outlinks.push_back(o);
        o->inlinks.push_back(in);
      }
    }
    for (int j = 0; j < op.outputs_size(); j++) {
      auto &out_var = op.outputs(j);
      for (int k = 0; k < out_var.arguments_size(); k++) {
        auto *out = graph->nodes.GetMutable(var2id[out_var.arguments(k)]);
        if (unique_written_vars.count(out)) {
          // Loop found, for example, a = op(a), use SSA, change to a1 = op(a).
          auto *out_alias = graph->nodes.Create(Node::Type::kValue);
          out_alias->SetName(out->name());
          out_alias->SetPbDesc(out->pb_desc());
          out_alias->SetPbMsg(out->pb_msg());
          var2id[out_alias->name()] =
              out_alias->id();  // update variable's alias Node
          LOG(INFO) << "loop found in graph, create SSA alias node ["
                    << out_alias->repr() << "] for [" << out->repr() << "]";
          out = out_alias;
        }
        out->inlinks.push_back(o);
        o->outlinks.push_back(out);
        unique_written_vars.insert(out);
      }
    }
  }
  // Analysis and extract the inputs and outputs of this graph.
  graph->Build();
}

namespace {
class DFG_DebuggerPass : public DFG_GraphvizDrawPass {
 public:
  using Config = DFG_GraphvizDrawPass::Config;
  explicit DFG_DebuggerPass(const Config &config)
      : DFG_GraphvizDrawPass(config) {}
  std::string repr() const override { return "fluid-to-dfg-debuger-pass"; }
  bool Finalize() override { return true; }
};
}

Pass *FluidToDataFlowGraphPass::CreateGraphvizDebugerPass() const {
  return new DFG_DebuggerPass(DFG_GraphvizDrawPass::Config(
      FLAGS_inference_analysis_graphviz_log_root, "fluid-to-dfg-debuger"));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
