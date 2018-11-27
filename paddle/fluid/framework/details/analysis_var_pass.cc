// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/analysis_var_pass.h"
#include <algorithm>
#include <atomic>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"

DEFINE_bool(enable_subgraph_optimize, false,
            "SubGraph also reuse global graph variables, it will reduce the "
            "memory occupation"
            "but a higher risk of memory reuse error. default disabled.");
DEFINE_string(memory_optimize_debug, "",
              "debug the operator output variable when do the variable reuse."
              "memory reuse pass."
              "only for debug, default disabled.");

namespace paddle {
namespace framework {
namespace details {

std::unordered_set<ir::Node*> AnalysisVarPass::GetSubBlockOutputVars(
    const std::unordered_set<ir::Node*>& nodes) const {
  std::unordered_set<ir::Node*> vars;
  std::unordered_map<std::string, ir::Node*> var_to_node_map;
  ProgramDesc* program = nullptr;
  for (auto& op : nodes) {
    if (op->IsOp() && op->Op() != nullptr && program == nullptr) {
      program = op->Op()->Block()->Program();
    }
    if (op->IsVar() && !op->IsCtrlVar()) {
      var_to_node_map[op->Name()] = op;
    }
  }
  if (program->Size() > 1) {
    // size > 1 means program has subblock. A subblock's AllVars
    // only contains the output variables, no input variables
    // are included.
    for (size_t i = 1; i < program->Size(); ++i) {
      auto& block_desc = program->Block(i);
      auto subblock_output_vars = block_desc.AllVars();
      for (auto& var_desc : subblock_output_vars) {
        vars.insert(var_to_node_map[var_desc->Name()]);
      }
    }
  }
  return vars;
}

std::unordered_set<std::string> AnalysisVarPass::GetSubBlockVars(
    const std::unordered_set<ir::Node*>& nodes) const {
  std::unordered_set<std::string> vars;
  for (auto& op : nodes) {
    if (!op->IsOp() || op->Op() == nullptr) continue;
    auto* op_desc = op->Op();
    if (OpHasSubBlock(op_desc)) {
      auto inputs = op_desc->InputArgumentNames();
      auto outputs = op_desc->OutputArgumentNames();
      vars.insert(inputs.begin(), inputs.end());
      vars.insert(outputs.begin(), outputs.end());
    }
  }
  return vars;
}

void AnalysisVarPass::RenameVarInGraphDesc(const std::string& var,
                                           const std::string& cache_var,
                                           size_t idx) const {
  for (size_t i = idx; i < cfg_.Ops().size(); ++i) {
    auto* op = cfg_.Ops()[i];
    auto* op_desc = op->Op();
    op_desc->Rename(var, cache_var);
    if (op_desc->Block()->HasVar(var)) {
      op_desc->Block()->RemoveVar(var);
    }

    FilterVariables(op->inputs, [&](ir::Node* node) {
      if (node->Name() == var) {
        node->Var()->SetName(cache_var);
      }
    });

    FilterVariables(op->outputs, [&](ir::Node* node) {
      if (node->Name() == var) {
        node->Var()->SetName(cache_var);
      }
    });
  }
}

void AnalysisVarPass::RenameVarInGraphNode(const std::string& var,
                                           const std::string& cache_var,
                                           size_t idx) const {
  for (size_t i = idx; i < cfg_.Ops().size(); ++i) {
    auto* op = cfg_.Ops()[i];

    FilterVariables(op->inputs, [&](ir::Node* node) {
      if (node->Name() == var) {
        node->SetName(cache_var);
      }
    });

    FilterVariables(op->outputs, [&](ir::Node* node) {
      if (node->Name() == var) {
        node->SetName(cache_var);
      }
    });

  }
}

bool AnalysisVarPass::NodeCanReused(ir::Node* node) const {
  if (!(node->IsVar() && !node->IsCtrlVar())) return false;
  auto* desc = node->Var();
  proto::VarType::Type type = desc->GetType();
  if (desc->Persistable() == true || type != proto::VarType::LOD_TENSOR ||
      desc->GetShape().size() == 0) {
    return false;
  }
  // vars can be @EMPTY@, @LR_DECAY_COUNTER@. For example, while_grad
  std::string name = node->Name();
  if (name.size() > 0 && name[0] == '@' && name[name.size() - 1] == '@')
    return false;
  if (skip_set_.count(name)) return false;
  for (auto* op : node->inputs) {
    if (op->Op()->HasAttr("force_cpu")) {
      // op output force generated in cpu, can not be reused.
      return framework::AttrReader(op->Op()->GetAttrMap())
                 .Get<bool>("force_cpu") != true;
    }
  }
  return true;
}

bool AnalysisVarPass::OpHasSubBlock(OpDesc* desc) const {
  const AttributeMap& attrs = desc->GetAttrMap();
  for (auto& attr : attrs) {
    if (attr.second.type() == typeid(BlockDesc*) ||             // NOLINT
        attr.second.type() == typeid(std::vector<BlockDesc*>))  // NOLINT
      return true;
  }
  return false;
}

std::unique_ptr<ir::Graph> AnalysisVarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& graph_pool = Get<GraphNodePool>(kGraphNodePool);
  auto nodes = graph->Nodes();
  auto subblock_output_vars = GetSubBlockOutputVars(nodes);
  auto subblock_vars = GetSubBlockVars(nodes);
  skip_set_.insert(subblock_vars.begin(), subblock_vars.end());

  cfg_ = details::ControlFlowGraph(*graph.get());
  cfg_.LiveVariableAnalysis();
  int counter = 0;
  for (size_t idx = 0; idx < cfg_.Ops().size(); ++idx) {
    auto& op = cfg_.Ops()[idx];
    auto* op_desc = op->Op();
    // some op in graph has no op desc
    if (op_desc == nullptr) continue;
    if (OpHasSubBlock(op_desc) && FLAGS_enable_subgraph_optimize) {
      // conditional block, while op and their grad op
      auto* sub_block_desc =
          AttrReader(op_desc->GetAttrMap()).Get<BlockDesc*>("sub_block");
      int sub_counter = 0;
      for (auto* sub_op_desc : sub_block_desc->AllOps()) {
        for (auto& sub_op_output_var_pair : sub_op_desc->Outputs()) {
          for (auto& sub_op_output_var : sub_op_output_var_pair.second) {
            auto* var_desc = sub_block_desc->FindVar(sub_op_output_var);
            ir::Node* var = ir::CreateNodeForTest(var_desc).get();
            if (NodeCanReused(var)) {
              ir::Node* cache = pool_.NodeMatch(var);
              if (cache != nullptr) {
                if (var->Var()->GetDataType() != cache->Var()->GetDataType()) {
                  continue;
                }
                int node_idx_in_pool = pool_.GetPosition(cache);
                VLOG(3) << string::Sprintf(
                    "!!! %s,  %s => %s, cache idx %d, pool size %d",
                    std::to_string(sub_counter), DebugString(var),
                    DebugString(cache), node_idx_in_pool,
                    static_cast<int>(pool_.size()));
                sub_counter += 1;
                // NOTE(dzh): subblock is not in IR graph. Modify the block_desc
                // immediately to make the subblock variable reuse strategy take
                // effect. Because it is a single op in graph. No need to update
                // the ir nodes.
                sub_op_desc->Rename(var->Name(), cache->Name());
                if (sub_op_desc->Block()->HasVar(var->Name())) {
                  sub_op_desc->Block()->RemoveVar(var->Name());
                }
              }
            }
          }
        }
      }
    } else {
      if (OpHasSubBlock(op_desc)) {
        VLOG(3) << op->Name()
                << " has subblock, but disable subgraph optimize. skipped.";
      }
    }

    for (auto& var : op->outputs) {
      if (NodeCanReused(var) && cfg_.Use(op).count(var->Name()) == 0) {
        ir::Node* cache = pool_.NodeMatch(var);
        if (FLAGS_memory_optimize_debug != "") {
          if (var->Name() == FLAGS_memory_optimize_debug) {
            VLOG(3) << "start match var " << DebugString(var) << " of op "
                    << op->Name();
            VLOG(3) << pool_.ToString();
            VLOG(3) << "matched in pool : "
                    << ((cache == nullptr) ? "False" : "True");
          }
        }
        if (cache != nullptr) {
          if (var->Name() == cache->Name()) {
            VLOG(3) << string::Sprintf("Same var!!! %s", DebugString(var));
            continue;
          }
          if (var->Var()->GetDataType() != cache->Var()->GetDataType()) {
            continue;
          }
          int node_idx_in_pool = pool_.GetPosition(cache);
          VLOG(3) << string::Sprintf(
              "!!! %s,  %s => %s, cache idx %d, pool size %d",
              std::to_string(counter), DebugString(var), DebugString(cache),
              node_idx_in_pool, static_cast<int>(pool_.size()));
          counter += 1;

          cfg_.RenameVarInCFGGraph(var->Name(), cache->Name(), idx);
          RenameVarInGraphDesc(var->Name(), cache->Name(), idx);
          RenameVarInGraphNode(var->Name(), cache->Name(), idx);
          pool_.Erase(cache);
        }
      }
    }
    // fill the pool
    for (auto var : cfg_.LiveIn(op)) {
      if (cfg_.LiveOut(op).count(var) == 0) {
        ir::Node* var_node = cfg_.GetNodeFromVarName(var, op);
        if (var_node == nullptr) continue;
        if (NodeCanReused(var_node) && !pool_.Has(var_node)) {
          pool_.Insert(var_node, op);
        }
      }
    }
  }

  // For early delete pass. use GraphNodePool load the unlived vars.
  // 1. find all deps op for each unlived var in memory pool.
  for (auto& op : cfg_.Ops()) {
    for (auto& var : op->inputs) {
      if (pool_.Has(var)) {
        pool_.Insert(var, op);
      }
    }
  }
  // 2. convert ir node based memory pool to graph node
  // because Node* maybe released bettwen passes.
  for (auto it = pool_.begin(); it != pool_.end(); ++it) {
    std::unordered_set<OpDesc*> descs;
    for (auto& op : it->second) {
      PADDLE_ENFORCE(op->IsOp());
      descs.insert(op->Op());
    }
    graph_pool.push_back(std::make_pair(it->first->Name(), descs));
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(analysis_var_pass, paddle::framework::details::AnalysisVarPass)
    .RequirePassAttr(paddle::framework::details::kGraphNodePool);
