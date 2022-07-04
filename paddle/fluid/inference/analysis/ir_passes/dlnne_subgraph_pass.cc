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
#include "paddle/fluid/inference/analysis/ir_passes/dlnne_subgraph_pass.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/dlnne_reg_py.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {

int (*PyConvertGraph)(const char *graph_name);

int RegisterPyFunc(const std::string &name, void *pfn) {
  if (name.compare("convert_graph") == 0) {
    PyConvertGraph = reinterpret_cast<decltype(PyConvertGraph)>(pfn);
  }

  return 0;
}
int ConvertGraph(std::string graph_name) {
  LOG(INFO) << "starting doing convert_graph";

  PyConvertGraph(graph_name.c_str());

  return 0;
}

namespace analysis {

using framework::ir::Node;

void analysis::DlnneSubgraphPass::ApplyImpl(framework::ir::Graph *graph) const {
  static std::unordered_set<std::string> teller_set{
      "mul",
      "matmul",
      "conv2d",
      "pool2d",
      "relu",
      "softmax",
      "sigmoid",
      "hard_swish",
      "depthwise_conv2d",
      "batch_norm",
      "concat",
      "tanh",
      "pad",
      "elementwise_add",
      "elementwise_mul",
      "dropout",
      "prelu",
      "conv2d_transpose",
      "leaky_relu",
      // "fc",
      "shuffle_channel",
      "swish",
      "split",
      // "instance_norm",
      "gelu",
      // "layer_norm",
      // "scale",
      // "stack",
      "relu6",
      "reshape2",
      "transpose2",
      "concat",
      "slice",
  };

  framework::ir::FusePassBase::Init("dlnne_subgraph_pass", graph);

  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    return teller_set.find(node->Op()->Type()) != teller_set.end();
  };

  framework::ir::SubGraphFuser fuser(
      graph,
      teller,
      Get<int>("min_subgraph_size") /*min subgraph size*/,
      "dlnne_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in dlnne, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      CreateDlnneOp(node, graph, graph_param_names, &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          framework::ir::Agent(node).subgraph()->begin(),
          framework::ir::Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += predictor_id;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}
std::string replace_name(std::string name,
                         const char *raw,
                         const char *new_char) {
  std::string r_name = name;
  int pos = r_name.find(raw);
  while (pos >= 0) {
    r_name = r_name.replace(pos, 1, new_char);
    pos = r_name.find(raw);
  }
  return r_name;
}

void DlnneSubgraphPass::CreateDlnneOp(
    framework::ir::Node *node,
    framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The subgraph should not be empty."));

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";
  // for debug
  framework::ProgramDesc tmp_dump_program_desc;
  auto *tmp_dump_main_block = tmp_dump_program_desc.MutableBlock(0);

  std::unordered_map<std::string, framework::VarDesc *> name_var_desc;
  std::set<std::string> name_var_input_nodes;
  std::set<std::string> name_var_output_nodes;
  std::set<std::string> name_ops;

  for (auto *node : subgraph) {
    auto *op = block_desc.AppendOp();
    *op->Proto() = *node->Op()->Proto();

    // debug
    {
      name_ops.insert(node->Name());
      auto *tmp_dump_new_block_op = tmp_dump_main_block->AppendOp();

      framework::OpDesc op_desc;
      op_desc.CopyFrom(*node->Op());

      for (auto argument_name : op_desc.InputArgumentNames()) {
        if (std::count(
                graph_params.begin(), graph_params.end(), argument_name) > 0) {
          op_desc.Rename(argument_name, replace_name(argument_name, "/", "."));
        }
      }
      for (auto argument_name : op_desc.OutputArgumentNames()) {
        if (std::count(
                graph_params.begin(), graph_params.end(), argument_name) > 0) {
          op_desc.Rename(argument_name, replace_name(argument_name, "/", "."));
        }
      }
      *tmp_dump_new_block_op->Proto() = *op_desc.Proto();

      for (auto *x : node->inputs) {
        if (x->IsVar()) {
          name_var_desc[x->Name()] = x->Var();
        }
        if (std::count(graph_params.begin(), graph_params.end(), x->Name()) ==
            0)
          name_var_input_nodes.insert(x->Name());
      }

      for (auto *x : node->outputs) {
        if (x->IsVar()) {
          name_var_desc[x->Name()] = x->Var();
        }
        if (std::count(graph_params.begin(), graph_params.end(), x->Name()) ==
            0)
          name_var_output_nodes.insert(x->Name());
      }
    }
  }
  std::set<std::string> valid_input_names;
  std::set<std::string> valid_output_names;
  for (auto name : name_var_output_nodes) {
    if (name_var_input_nodes.find(name) == name_var_input_nodes.end()) {
      valid_output_names.insert(name);
    }
  }

  for (auto name : name_var_input_nodes) {
    if (name_var_output_nodes.find(name) == name_var_output_nodes.end()) {
      valid_input_names.insert(name);
    }
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  // if we delete fluid copy of params shared by more than 1 ops, there will be
  // problem, so we filter them out.

  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
  }

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  std::vector<int> origin_output_dims;
  for (auto *x : node->outputs) {
    origin_output_dims.push_back(x->Var()->GetShape().size());
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }

  std::unordered_map<std::string, std::string> output_name_map;
  std::unordered_map<std::string, framework::ir::Node *> graph_var_map;

  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      graph_var_map[node->Name()] = node;
    }
  }

  // Set attrs
  op_desc->SetType("dlnne_engine");
  op_desc->SetInput("Xs",
                    std::vector<std::string>(valid_input_names.begin(),
                                             valid_input_names.end()));

  op_desc->SetOutput("Ys",
                     std::vector<std::string>(valid_output_names.begin(),
                                              valid_output_names.end()));

  op_desc->SetAttr("parameters", params);
  auto engine_key = GenerateEngineKey(
      input_names_with_id, output_names_with_id, std::to_string(0));
  op_desc->SetAttr("engine_key", engine_key);
  auto *scope = param_scope();

  {
    std::set<std::string> input_names;

    for (auto name : name_var_input_nodes) {
      if (name_var_output_nodes.find(name) == name_var_output_nodes.end()) {
        input_names.insert(name);
      }
    }

    // add feed to subgraph:
    int input_idx = 0;
    for (auto input_name : input_names) {
      auto *feed0 = tmp_dump_main_block->AppendOp();
      feed0->SetType("feed");
      feed0->SetInput("X", {"feed"});
      feed0->SetOutput("Out", {input_name});
      feed0->SetAttr("col", input_idx);
      input_idx++;
    }
    // add fetch to subgraph:
    int output_idx = 0;
    for (auto output_name : valid_output_names) {
      auto *fetch0 = tmp_dump_main_block->AppendOp();
      fetch0->SetType("fetch");
      fetch0->SetInput("X", {output_name});
      fetch0->SetOutput("Out", {"out"});
      fetch0->SetAttr("col", output_idx);
      output_idx++;
    }

    mkdir("./dump", 0777);
    std::string dir_name = "./dump/" + engine_key;
    mkdir(dir_name.c_str(), 0777);
    ofstream m_stream;
    m_stream.open(dir_name + "/__model__", ios::out);

    VLOG(4) << "name_var_desc size:" << name_var_desc.size();

    for (auto &kv : name_var_desc) {
      auto *new_add_var = tmp_dump_main_block->Proto()->add_vars();
      *new_add_var = *kv.second->Proto();
      auto *variable_tmp = scope->FindVar(kv.first);
      if (variable_tmp != nullptr) {
        *new_add_var->mutable_name() = replace_name(kv.first, "/", ".");
        new_add_var->set_persistable(true);
      } else {
        new_add_var->set_persistable(false);
      }
    }

    for (auto param_name : params) {
      auto *var = scope->FindVar(param_name);
      if (var != nullptr) {
        auto *var_t = var->GetMutable<framework::LoDTensor>();
        ofstream p_stream;
        p_stream.open(dir_name + "/" + replace_name(param_name, "/", "."),
                      ios::out);
        platform::DeviceContextPool &pool =
            platform::DeviceContextPool::Instance();
        auto &dev_ctx = *pool.Get(var_t->place());
        framework::SerializeToStream(p_stream, *var_t, dev_ctx);
        p_stream.close();
      }
    }

    std::string model;

    tmp_dump_program_desc.Proto()->SerializeToString(&model);
    m_stream << model;
    m_stream.close();

    op_desc->SetBlockAttr("sub_block", tmp_dump_main_block);
    op_desc->SetAttr("subgraph", model);
    op_desc->Flush();

    ConvertGraph(engine_key);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(dlnne_subgraph_pass,
              paddle::inference::analysis::DlnneSubgraphPass);
