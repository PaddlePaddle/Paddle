// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <iostream>
#include <fstream>

#include "paddle/fluid/inference/lite/op_teller.h"
#include "paddle/fluid/framework/lod_tensor.h"

#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/string/pretty_log.h"

#include "paddle/fluid/inference/lite/engine.h"

#include "google/protobuf/text_format.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

std::vector<std::string> IOVarsFilter(const std::vector<Node*>& nodes) {
  std::vector<std::string> names;
  for (const auto& node: nodes) {
    if (node->IsVar() && !node->Var()->Persistable()) { 
      names.push_back(node->Name());
    }
  }
  return names;
}

void StrToBinaryFile(const std::string& path, const std::string& str) {
  std::ofstream file(path.c_str(), std::ios::binary);
  file.write(str.c_str(), str.size());
  file.close();
}

void LiteSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {

  framework::ir::FusePassBase::Init("lite_subgraph_pass", graph);
  framework::ProgramDesc* global_program = Get<framework::ProgramDesc *>("program");

  // auto &lite_ops_filter = Get<std::vector<std::string>>("lite_ops_filter");
  std::vector<std::string> lite_ops_filter = {};

  auto teller = [&lite_ops_filter](const Node *node) {
    if (!node->IsOp() || !node->Op())
      return false;
    else if (std::find(lite_ops_filter.begin(), lite_ops_filter.end(),
                       node->Op()->Type()) != lite_ops_filter.end())
      return false;
    return lite::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph, teller, 0 /* min_subgraph_size */, "lite_engine");
  fuser();

  std::vector<std::string> repetitive_params;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      BuildOperator(node, global_program, &repetitive_params);
      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

void LiteSubgraphPass::BuildOperator(
    Node *merged_node, framework::ProgramDesc* global_program,
    std::vector<std::string> *repetitive_params) const {
  
  framework::ProgramDesc engine_program;

  OrganizeProgram(merged_node, global_program, &engine_program, repetitive_params);
  //SetUpEngine(&engine_program, repetitive_params);

  auto *op_desc = merged_node->Op();
  op_desc->SetInput("Xs", IOVarsFilter(merged_node->inputs));
  op_desc->SetOutput("Ys", IOVarsFilter(merged_node->outputs));
  op_desc->SetType("lite_engine");
  op_desc->SetAttr("engine_key", std::string("engine_key"));
}

// The modification of pass should be a process of framework::desc
// (initial) -> proto::desc (flush) -> framework::desc (final).
// Ir::Graph is limited to changing the main block, so the sub block
// needs to be processed here.
void LiteSubgraphPass::OrganizeProgram(Node *merged_node,
  framework::ProgramDesc* host_program,
  framework::ProgramDesc* engine_program,
  std::vector<std::string> *repetitive_params) const {
  std::vector<framework::ir::Node *>& subgraph = *Agent(merged_node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());
  const framework::BlockDesc &host_global_block = host_program->Block(framework::kRootBlockIndex);
  framework::BlockDesc* host_sub_block = host_program->AppendBlock(host_global_block);

  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  std::unordered_set<Node *> io_var_nodes = GetRelatedIOVarNodes(subgraph);
  *repetitive_params = ExtractParameters(io_var_nodes);

  std::vector<framework::OpDesc*> subgraph_ops;
  for (auto *op_node : subgraph) {
    subgraph_ops.push_back(op_node->Op());
  }

  ModifyHostProgram(host_program, host_sub_block, io_var_nodes, subgraph_ops);

  ModifyEngineProgram(merged_node, host_program, engine_program, host_sub_block,
                      io_var_nodes, subgraph_ops);

  host_program->Flush();
  engine_program->Flush();

  std::string str;
  google::protobuf::TextFormat::PrintToString(*engine_program->Proto(), &str);
  std::cout << "=====" << std::endl;
  std::cout << str;
}

void LiteSubgraphPass::SetUpEngine(framework::ProgramDesc* program,
  std::vector<std::string> *repetitive_params) const {
  inference::lite::EngineConfig config;
  auto *scope = param_scope();

  auto serialize_params = [] (std::string* str, framework::Scope* scope,
       const std::vector<std::string>& params) {
    std::ostringstream os;
    platform::CPUDeviceContext ctx;
    for (const auto& param: params) {
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    *str = os.str();
  };

  serialize_params(&config.param, scope, *repetitive_params);
  config.model = program->Proto()->SerializeAsString();
  StrToBinaryFile("./model.bin", config.model);
  config.prefer_place = paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)});
  config.valid_places = {
      paddle::lite::Place({TARGET(kHost), PRECISION(kFloat)}),
      paddle::lite::Place({TARGET(kCUDA), PRECISION(kFloat)}),
  };
  inference::Singleton<inference::lite::EngineManager>::Global()
      .Create("engine_key", config);
}

void LiteSubgraphPass::ModifyHostProgram(framework::ProgramDesc* host_program,
  framework::BlockDesc* host_sub_block,
  const std::unordered_set<Node *>& io_var_nodes,
  const std::vector<framework::OpDesc*>& subgraph_ops) const {
  for (auto *var_node: io_var_nodes) {
    auto* sub_block_var = host_sub_block->Var(var_node->Name());
    sub_block_var->Proto()->CopyFrom(*var_node->Var()->Proto());
  }
  for (auto *op_desc : subgraph_ops) {
    auto* sub_block_op = host_sub_block->AppendOp();
    sub_block_op->Proto()->CopyFrom(*op_desc->Proto());
    if (op_desc->HasAttr("sub_block")) {
      int32_t global_sub_id = host_sub_block->ID();
      auto *op_sub_block = host_program->MutableBlock(op_desc->GetBlockAttrId("sub_block"));
      op_sub_block->Proto()->set_parent_idx(global_sub_id);
    }
  }
}

void LiteSubgraphPass::ModifyEngineProgram(Node *merged_node,
  framework::ProgramDesc* host_program,
  framework::ProgramDesc* engine_program,
  framework::BlockDesc* host_sub_block,
  const std::unordered_set<Node *>& io_var_nodes,
  const std::vector<framework::OpDesc*>& subgraph_ops) const {

  // 1. Fill the main block of lite program.
  framework::BlockDesc* engine_global_block = engine_program->MutableBlock(framework::kRootBlockIndex);
  PrependFeedOps(engine_global_block, IOVarsFilter(merged_node->inputs));
  for (auto *var_node: io_var_nodes) {
    framework::VarDesc* sub_block_var = engine_global_block->Var(var_node->Name());
    sub_block_var->Proto()->CopyFrom(*var_node->Var()->Proto());
  }
  PrependFetchOps(engine_global_block, IOVarsFilter(merged_node->outputs));

  // 2. Append sub blocks in the lite program.
  std::unordered_map<int32_t, int32_t> sub_blocks_map;
  std::unordered_set<int32_t> copied_host_ids;
  sub_blocks_map[host_sub_block->ID()] = framework::kRootBlockIndex;
  std::function<void(const std::vector<framework::OpDesc*>&)> append_sub_blocks;
  append_sub_blocks = [&](const std::vector<framework::OpDesc*>& ops) {
    for (auto *op_desc : ops) {
      if (op_desc->HasAttr("sub_block")) {
        int32_t host_op_sub_id = op_desc->GetBlockAttrId("sub_block");
        if (copied_host_ids.count(host_op_sub_id)) continue;
        size_t engine_block_size = engine_program->Size();
        auto* host_op_sub_block = host_program->MutableBlock(host_op_sub_id);
        auto* engine_op_sub_block = engine_program->AppendBlock(*(op_desc->Block()));
        for (auto* var: host_op_sub_block->AllVars()) {
          auto* engine_var = engine_op_sub_block->Var(var->Name());
          engine_var->Proto()->CopyFrom(*var->Proto());
        }
        for (auto* op: host_op_sub_block->AllOps()) {
          auto* engine_op = engine_op_sub_block->AppendOp();
          engine_op->Proto()->CopyFrom(*op->Proto());
        }
        sub_blocks_map[host_op_sub_id] = engine_block_size;
        append_sub_blocks(host_op_sub_block->AllOps());
      }
    }
  };
  append_sub_blocks(subgraph_ops);
  for (size_t i = 0; i < engine_program->Size(); i++) {
    for (auto *op_desc : engine_program->Block(i).AllOps()) {
      if (op_desc->HasAttr("sub_block")) {
        int32_t id = op_desc->GetBlockAttrId("sub_block");
        op_desc->SetAttr("sub_block", sub_blocks_map[id]);
      }
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(lite_subgraph_pass,
              paddle::inference::analysis::LiteSubgraphPass);
