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

#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimize_pass.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void MemoryOptimizePass::ApplyImpl(ir::Graph* graph) const {
  CollectSkipVarsSet(graph);

  cfg_.reset(new ControlFlowGraph(*graph));
  cfg_->LiveVariableAnalysis();
  InitSSAGraphNodes();

  int reuse_id = 0;
  for (size_t idx = 0; idx < cfg_->Ops().size(); ++idx) {
    auto& op = cfg_->Ops()[idx];
    auto* op_desc = op->Op();
    // some op in graph has no op desc
    if (op_desc == nullptr) continue;

    for (auto& var : op->outputs) {
      if (var->IsVar() && !var->IsCtrlVar() && skip_set_.count(var->Name())) {
        VLOG(3) << "Skip set contains variable of " << var->Name()
                << "disable reuse on it. skipped";
        continue;
      }
      if (NodeCanReused(var) && cfg_->Use(op).count(var->Name()) == 0) {
        ir::Node* cache = pool_.FindBestFitNode(var);
        while (cache != nullptr && var->Name() == cache->Name()) {
          VLOG(3) << "The same cache variable is cascade reused. "
                  << cache->Name() << " is re-filled to the pool after "
                  << "the reused op is finished. Current op can not "
                  << "replace it again. Skip this candidate.";
          cache = pool_.FindNextBestFitNode(var, cache);
        }

        if (cache != nullptr) {
          int node_idx_in_pool = pool_.GetNodeIndexInPool(cache);
          VLOG(3) << string::Sprintf(
              "!!! %s,  %s => %s, cache idx %d, pool size %d",
              std::to_string(reuse_id++), DebugString(var), DebugString(cache),
              node_idx_in_pool, static_cast<int>(pool_.size()));
          // NOTE(dzhwinter): update the ProgramDesc/IR Graph
          // and the CFG Graph on the fly.
          //
          // IR Graph define the dependence relationship between nodes.
          //
          // ProgramDesc defines the input/output vars. Its used in
          // CreateOp, CreateVar when running happens.
          //
          // CFG Graph store the liveness information, when reuse happens
          // we also need to update the variable liveness.
          const std::string var_name = var->Name();
          const std::string cache_name = cache->Name();

          cfg_->RenameVarInCFGGraph(var_name, cache_name, idx);
          RenameVarInGraphDesc(var_name, cache_name, idx);
          RenameVarInGraphNode(var_name, cache_name, idx, graph);
          pool_.Erase(cache_name);
        }
      }
    }
    // fill the pool
    for (auto& var : cfg_->Unlived(op)) {
      ir::Node* var_node = cfg_->GetNodeByName(var, op);
      if (var_node == nullptr || var_node->IsCtrlVar()) continue;
      if (NodeCanReused(var_node) && !pool_.Has(var_node)) {
        pool_.Insert(var_node);
      }
    }
  }
  graph->ResolveHazard(var_nodes_);
}

void MemoryOptimizePass::CollectSkipVarsSet(ir::Graph* graph) const {
  // fill skip_set_
  PADDLE_ENFORCE(graph->Has(kMemOptSkipVars));
  auto& mem_opt_whitelist = graph->Get<MemOptSkipVars>(kMemOptSkipVars);
  for (const auto& var : mem_opt_whitelist) {
    skip_set_.emplace(var);
  }
}

void MemoryOptimizePass::RenameVarInGraphDesc(const std::string& var,
                                              const std::string& cache_var,
                                              size_t idx) const {
  for (size_t i = idx; i < cfg_->Ops().size(); ++i) {
    auto* op = cfg_->Ops()[i];
    PADDLE_ENFORCE(op->IsOp() && op->Op());
    auto* op_desc = op->Op();
    op_desc->RenameInput(var, cache_var);
    op_desc->RenameOutput(var, cache_var);
    if (op_desc->Block() != nullptr) {
      op_desc->Block()->RemoveVar(var);
    } else {
      LOG(WARNING) << "op " << op->Name() << " not know its block."
                   << "Is the op_desc created without block pointer? "
                   << "Can not find " << var << " in Block(0)";
    }
    op_desc->Flush();
  }
}

void MemoryOptimizePass::InitSSAGraphNodes() const {
  std::unordered_map<std::string, std::unordered_set<ir::Node*>> all_vars;
  if (var_nodes_.empty()) {
    for (auto* op : cfg_->Ops()) {
      for (auto* node : op->inputs) {
        if (all_vars[node->Name()].count(node) == 0) {
          all_vars[node->Name()].emplace(node);
          var_nodes_[node->Name()].emplace_back(node);
        }
      }
      for (auto* node : op->outputs) {
        if (all_vars[node->Name()].count(node) == 0) {
          all_vars[node->Name()].emplace(node);
          var_nodes_[node->Name()].emplace_back(node);
        }
      }
    }
  }
}

void MemoryOptimizePass::RenameVarInGraphNode(const std::string& var,
                                              const std::string& cache_var,
                                              size_t idx,
                                              ir::Graph* graph) const {
  // if replace happens, we need to create a newer version cache_var
  // but use the same dims/data_type with var.
  PADDLE_ENFORCE(var_nodes_[var].size() >= 1 &&
                 var_nodes_[var].at(0)->Var() != nullptr);
  std::unique_ptr<VarDesc> var_desc(new VarDesc(*var_nodes_[var].at(0)->Var()));
  var_desc->SetName(cache_var);

  for (size_t i = idx; i < cfg_->Ops().size(); ++i) {
    auto* op = cfg_->Ops()[i];

    // redirect the input to the latest version of cache_var
    for (auto* node : op->inputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = var_nodes_[cache_var].back();

        // swap node to cache_node
        cache_node->outputs.insert(cache_node->outputs.end(),
                                   node->outputs.begin(), node->outputs.end());
        PADDLE_ENFORCE(node->inputs.size() == 1 && node->inputs[0]->IsOp());
        auto* prev_op = node->inputs[0];
        std::replace(prev_op->outputs.begin(), prev_op->outputs.end(), node,
                     cache_node);
        for (auto* next_op : node->outputs) {
          std::replace(next_op->inputs.begin(), next_op->inputs.end(), node,
                       cache_node);
        }

        // erase unused node
        auto& nodes = var_nodes_.at(var);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
        graph->RemoveNode(node);
      }
    }

    // if we need to rename the output,
    // always create a newer version of cache_var
    for (auto* node : op->outputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = graph->CreateVarNode(var_desc.get());
        var_nodes_[cache_var].emplace_back(cache_node);

        // swap node to cache node
        cache_node->outputs.insert(cache_node->outputs.end(),
                                   node->outputs.begin(), node->outputs.end());
        cache_node->inputs.emplace_back(op);
        std::replace(op->outputs.begin(), op->outputs.end(), node, cache_node);
        for (auto* next_op : node->outputs) {
          std::replace(next_op->inputs.begin(), next_op->inputs.end(), node,
                       cache_node);
        }

        // erase unused node
        auto& nodes = var_nodes_.at(var);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
        graph->RemoveNode(node);
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(memory_optimize_pass, paddle::framework::ir::MemoryOptimizePass)
    .RequireGraphAttr(paddle::framework::details::kStaleProgramOpDescs);
