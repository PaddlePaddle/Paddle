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
DEFINE_bool(memory_optimize_debug, true,
            "Swap node immediately without another"
            "memory reuse pass."
            "only for debug, default disabled.");

namespace paddle {
namespace framework {
namespace details {
using details::UnlivedNodePool;
using details::ReusedNodePairMap;
using details::ControlFlowGraph;

template <typename Container>
std::string PrintName(const AnalysisVarPass* pass, const Container& cons) {
  std::vector<std::string> ans;
  for (auto& item : cons) {
    // ans.push_back(item.first->Name());
    ans.push_back(item);
  }
  std::sort(ans.begin(), ans.end());
  std::stringstream ss;
  for (auto& item : ans) {
    // ss << pass->DebugString(item.first) << " ";
    // ss << item.first->Name() << " ";
    ss << item << " ";
  }
  ss << std::endl;
  return ss.str();
}

const std::string AnalysisVarPass::DebugString(ir::Node* var) const {
  std::stringstream ss;
  ss << var->Name();
  // ss << ":" << DataTypeToString(var->Var()->GetDataType());
  ss << "[";
  try {
    auto shape = var->Var()->GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != shape.size() - 1) {
        ss << shape[i] << ",";
      } else {
        ss << shape[i];
      }
    }
    ss << "]";
  } catch (...) {
    ss << "Var has no VarDesc !!! Name:" << var->Name()
       << " Type:" << var->IsOp();
  }
  return ss.str();
}

// bool AnalysisVarPass::NodeMatch(ir::Node* var, ir::Node** cache,
//                                 int* node_idx_in_pool) const {
// }

std::unordered_set<ir::Node*> AnalysisVarPass::GetSubBlockOutputVars(
    const std::unordered_set<ir::Node*>& nodes) const {
  std::unordered_set<ir::Node*> vars;
  std::unordered_map<std::string, ir::Node*> var_to_node_map;
  ProgramDesc* program = nullptr;
  for (auto& op : nodes) {
    if (op->IsOp() && program == nullptr) {
      program = op->Op()->Block()->Program();
    }
    if (op->IsVar() && !op->IsCtrlVar()) {
      var_to_node_map[op->Name()] = op;
    }
  }
  if (program->Size() > 1) {
    // size>1 means program has subblock. A subblock's AllVars
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
    if (!op->IsOp()) continue;
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

void AnalysisVarPass::UpdateGraphAndDesc(size_t idx, ir::Node* var,
                                         ir::Node* cache_var) const {
  for (size_t i = idx; i < cfg.Ops().size(); ++i) {
    auto* op = cfg.Ops()[i];
    auto* op_desc = op->Op();
    // update desc
    op_desc->Rename(var->Name(), cache_var->Name());
    if (op_desc->Block()->HasVar(var->Name())) {
      op_desc->Block()->RemoveVar(var->Name());
    }
    // update node
    std::replace(op->inputs.begin(), op->inputs.end(), var, cache_var);
    std::replace(op->outputs.begin(), op->outputs.end(), var, cache_var);
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
  auto& g_pool = Get<UnlivedNodePool>(kUnlivedNodePool);
  // auto& node_pair_map = Get<ReusedNodePairMap>(kReusedNodePairMap);
  // auto& graph_ops = Get<GraphOpsReused>(kGraphOpsReused);
  // auto& graph_early_delete_op_deps =
  // Get<GraphEarlyDeleteOpsDeps>(kGraphEarlyDeleteOpsDeps);
  auto nodes = graph->Nodes();
  auto subblock_output_vars = GetSubBlockOutputVars(nodes);
  auto subblock_vars = GetSubBlockVars(nodes);
  VLOG(3) << PrintName(this, subblock_vars);

  // FillOpSignatureId(nodes);
  // cfg.reset(new details::ControlFlowGraph(*graph.get()));
  cfg = details::ControlFlowGraph(*graph.get());
  cfg.LiveVariableAnalysis();
  int counter = 0;
  for (size_t idx = 0; idx < cfg.Ops().size(); ++idx) {
    auto& op = cfg.Ops()[idx];

    // VLOG(3) << "live in " << cfg.LiveIn(op).size();
    // VLOG(3) << "live out " << cfg.LiveOut(op).size();
    // VLOG(3) << "use " << cfg.Use(op).size();
    // VLOG(3) << "def " << cfg.Def(op).size();
    // VLOG(3) << PrintSet(this, cfg.LiveIn(op));
    // VLOG(3) << PrintSet(this, cfg.LiveOut(op));
    // VLOG(3) << idx << "pool size : " << pool.size();
    // VLOG(3) << PrintName(this, pool);
    // VLOG(3) << PrintIt(this, cfg.LiveIn(op));
    // VLOG(3) << PrintIt(this, cfg.LiveOut(op));

    auto* op_desc = op->Op();
    if (OpHasSubBlock(op_desc)) {
      // if (FLAGS_enable_subgraph_optimize) {
      //   // conditional block, while op
      //   auto* sub_block_desc =
      //       AttrReader(op_desc->GetAttrMap()).Get<BlockDesc*>("sub_block");
      //   for (auto* sub_op_desc : sub_block_desc->AllOps()) {
      //     for (auto& sub_op_output_var_pair : sub_op_desc->Outputs()) {
      //       for (auto& sub_op_output_var : sub_op_output_var_pair.second) {
      //         auto* var_desc = sub_block_desc->FindVar(sub_op_output_var);
      //         ir::Node* var = ir::CreateDummyNode(var_desc).get();
      //         if (NodeCanReused(var)) {
      //           int node_idx_in_pool = -1;
      //           ir::Node* cached_var = nullptr;
      //           if (NodeMatch(var, &cached_var, &node_idx_in_pool)) {
      //             VLOG(3) << string::Sprintf(
      //                 "cache idx %d, pool size %d, var is %s, cached "
      //                 "var %s",
      //                 node_idx_in_pool, static_cast<int>(pool.size()),
      //                 DebugString(var), DebugString(cached_var));
      //             // pool.erase(cached_var);
      //             // subblock is not in IR graph. Modify the block_desc
      //             // immediately
      //             // to make the subblock variable reuse strategy take
      //             effect.
      //             sub_op_desc->Rename(var->Name(), cached_var->Name());
      //             if (sub_op_desc->Block()->HasVar(var->Name())) {
      //               sub_op_desc->Block()->RemoveVar(var->Name());
      //             }
      //           }
      //         }
      //       }
      //     }
      //   }
      // } else {
      //   VLOG(3) << op->Name()
      //           << " has subblock, but disable subgraph optimize. skipped.";
      //   continue;
      // }
      VLOG(3) << op->Name()
              << " has subblock, but disable subgraph optimize. skipped.";
      continue;
    }

    for (auto& var : op->outputs) {
      if (NodeCanReused(var) && subblock_vars.count(var->Name()) == 0 &&
          cfg.Use(op).count(var->Name()) == 0) {
        ir::Node* cache = pool.NodeMatch(var);

        if (cache != nullptr) {
          if (var->Name() == cache->Name()) {
            VLOG(3) << string::Sprintf("Same var!!! %s", DebugString(var));
            continue;
          }
          if (var->Var()->GetDataType() != var->Var()->GetDataType()) {
            continue;
          }
          int node_idx_in_pool = pool.GetPosition(cache);
          VLOG(3) << string::Sprintf(
              "!!! %s,  %s => %s, cache idx %d, pool size %d",
              std::to_string(counter), DebugString(var), DebugString(cache),
              node_idx_in_pool, static_cast<int>(pool.size()));
          counter += 1;
          UpdateGraphAndDesc(idx, var, cache);
          // if (FLAGS_memory_optimize_debug) {
          //   UpdateGraphAndDesc(idx, var, cache);
          // } else {
          //   node_pair_map[op->Op()->GetOrderId()] =
          //       std::make_pair(var->Name(), cache->Name());
          // }
          pool.Erase(cache);
          cfg.UpdateGraph(var->Name(), cache->Name(), idx);
        }
      }
    }

    for (auto var : cfg.LiveIn(op)) {
      if (cfg.LiveOut(op).count(var) == 0) {
        ir::Node* var_node = cfg.GetNodeFromVarName(var, op);
        if (var_node == nullptr) continue;
        if (NodeCanReused(var_node) && !pool.Has(var_node)) {
          pool.Insert(var_node, op);
        }
      }
    }
  }

  // fill pool with op order_id and name.
  for (auto& pair : pool) {
    g_pool.push_back(
        std::make_pair(pair.first->Name(), pair.second->Op()->GetOrderId()));
  }

  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(analysis_var_pass, paddle::framework::details::AnalysisVarPass)
    .RequirePassAttr(paddle::framework::details::kReusedNodePairMap)
    .RequirePassAttr(paddle::framework::details::kGraphOpsReused)
    .RequirePassAttr(paddle::framework::details::kUnlivedNodePool)
    .RequirePassAttr(paddle::framework::details::kGraphEarlyDeleteOpsDeps);
