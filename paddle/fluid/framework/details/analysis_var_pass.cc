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

#include <iostream>
#include <iterator>
#include <memory>
#include <atomic>
#include <sstream>
#include <type_traits>
#include <vector>
#include <algorithm>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/details/analysis_var_pass.h"

DEFINE_bool(enable_subgraph_optimize, false,
            "SubGraph also reuse global graph variables, it will reduce the "
            "memory occupation"
            "but a higher risk of memory reuse error. default disabled.");

namespace paddle {
namespace framework {
namespace details {
using details::UnlivedNodePool;
using details::ReusedNodePairMap;
using details::ControlFlowGraph;

template <typename Container>
std::string PrintIt(const AnalysisVarPass* pass, const Container& cons) {
  std::stringstream ss;
  for (auto& item : cons) {
    // ss << pass->DebugString(item.first) << " ";
    ss << pass->DebugString(item.first) << " ";
  }
  ss << std::endl;
  return ss.str();
}

int VarInBytes(ir::Node* var) {
  auto* desc = var->Var();
  auto shape = desc->GetShape();
  size_t type_size =
    framework::SizeOfType(framework::ToTypeIndex(desc->GetDataType()));
  int size = 1;
  for(auto& s: shape) { size *= s; }
  return type_size * std::abs(size);
}

// const UnlivedNodePool::iterator FindPosition(const UnlivedNodePool& pool, ir::Node* node) {
//   auto it = pool.begin(), pos = pool.begin();
//   while(it != pool.end()) {
//     auto* var = it->first;
//     if (VarInBytes(var) < VarInBytes(node)) {
//       pos++;
//     }
//     it++;
//   }
//   return pos;
// }

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

bool AnalysisVarPass::NodeMatch(ir::Node* var, ir::Node** cache,
                                int* node_idx_in_pool) const {
  auto get_node_size = [&](ir::Node* n) {
    auto* desc = n->Var();
    auto shape = desc->GetShape();
    size_t type_size =
        framework::SizeOfType(framework::ToTypeIndex(desc->GetDataType()));
    int size = 1;
    for(auto& s: shape) { size *= s; }
    return type_size * std::abs(size);
  };

  auto compare_node_size = [&](ir::Node* lhs, ir::Node* rhs) {
    auto* lhs_desc = lhs->Var();
    auto* rhs_desc = rhs->Var();
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();

    // -1 means shape is determined in runtime.
    if ((lhs_shape[0] == -1) ^ (rhs_shape[0] == -1)) return false;
    return get_node_size(lhs) <= get_node_size(rhs);
  };

  // linear search in an sorted node set, find the best fit node.
  for (auto it = pool.begin(); it != pool.end(); ++it) {
    ir::Node* cache_var = it->first;
    if (compare_node_size(var, cache_var)) {
      *cache = cache_var;
      *node_idx_in_pool = std::distance(pool.begin(), it);
      return true;
    }
  }
  return false;
}


std::unordered_set<ir::Node*> AnalysisVarPass::GetSubBlockOutputVars(
                                                                     const std::unoredered_set<ir::Node*>& nodes) const {
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

std::unique_ptr<ir::Graph> FillOpSignatureId(std::unique_ptr<ir::Graph> graph) {
  // NOTE(): the Graph/Node may changed in different passes. e.g. Mutildevice pass
  // will release origin IR nodes, merge batch pass will copy the OpDesc mutiple times.
  // We need a mark to record the reused op, the early deleted op dependencies ops.
  std::atomic<int> order_id(-1);
  for (auto& n : graph->Nodes()) {
    if (!n->IsOp()) continue;
    auto* op_desc = n->Op();
    op_desc->SetOrderId(order_id.fetch_add(1));
  }
}

std::unique_ptr<ir::Graph> AnalysisVarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto& g_pool = Get<UnlivedNodePool>(kUnlivedNodePool);
  auto& node_pair_map = Get<ReusedNodePairMap>(kReusedNodePairMap);
  auto& graph_ops = Get<GraphOpsReused>(kGraphOpsReused);
  auto& graph_early_delete_upstream_ops = Get<GraphEarlyDeleteOpsDeps>(kGraphEarlyDeleteOpsDeps);
  auto nodes = graph->Nodes();
  auto subblock_output_vars = etSubBlockOutputVars(nodes);

  auto op_has_subblock = [&](OpDesc* desc) {
    const AttributeMap& attrs = desc->GetAttrMap();
    for (auto& attr : attrs) {
      if (typeid(attr.second) == typeid(BlockDesc)) return true;
    }
    return false;
  };

  auto var_can_reused = [&](ir::Node* node) {
    PADDLE_ENFORCE(
        node->IsVar() && !node->IsCtrlVar(),
        string::Sprintf("Expect node %s as Variable.", node->Name()));
    auto* desc = node->Var();
    proto::VarType::Type type = desc->GetType();
    if (desc->Persistable() || type != proto::VarType::LOD_TENSOR ||
        desc->GetShape().size() == 0) {
      return false;
    }
    // Operation output vars can be @EMPTY@. For example, while_grad
    if (node->Name() == "@EMPTY@") return false;
    // op output force generated in cpu, instead of inferenced from input.
    for (auto* op : node->inputs) {
      if (op->Name() == "fill_constant" && op->Op()->HasAttr("force_cpu")) {
        return framework::AttrReader(op->Op()->GetAttrMap())
                   .Get<bool>("force_cpu") != true;
      }
    }
    return true;
  };

  FillOpSignatureId(graph);
  ControlFlowGraph cfg(*graph.get());
  cfg.LiveVariableAnalysis();
  for (size_t idx = 0; idx < cfg.Ops().size(); ++idx) {
    auto& op = cfg.Ops()[idx];

    VLOG(3) << PrintIt(this, pool);
    VLOG(3) << op->Name();
    // VLOG(3) << PrintIt(this, cfg.LiveIn(op));
    // VLOG(3) << PrintIt(this, cfg.LiveOut(op));

    auto* op_desc = op->Op();
    if (op_has_subblock(op_desc)) {
      if (FLAGS_enable_subgraph_optimize) {
        // conditional block, while op
        auto* sub_block_desc =
            AttrReader(op_desc->GetAttrMap()).Get<BlockDesc*>("sub_block");
        for (auto* sub_op_desc : sub_block_desc->AllOps()) {
          for (auto& sub_op_output_var_pair : sub_op_desc->Outputs()) {
            for (auto& sub_op_output_var : sub_op_output_var_pair.second) {
              auto* var_desc = sub_block_desc->FindVar(sub_op_output_var);
              ir::Node* var = ir::CreateDummyNode(var_desc).get();
              if (var_can_reused(var)) {
                int node_idx_in_pool = -1;
                ir::Node* cached_var = nullptr;
                if (NodeMatch(var, &cached_var, &node_idx_in_pool)) {
                  VLOG(3) << string::Sprintf(
                      "cache idx %d, pool size %d, var is %s, cached "
                      "var %s",
                      node_idx_in_pool, static_cast<int>(pool.size()), DebugString(var),
                      DebugString(cached_var));
                  // pool.erase(cached_var);
                  // subblock is not in IR graph. Modify the block_desc
                  // immediately
                  // to make the subblock variable reuse strategy take effect.
                  sub_op_desc->Rename(var->Name(), cached_var->Name());
                  if (sub_op_desc->Block()->HasVar(var->Name())) {
                    sub_op_desc->Block()->RemoveVar(var->Name());
                  }
                }
              }
            }
          }
        }
      } else {
        VLOG(3) << op->Name()
                << " has subblock, but disable subgraph optimize. skipped.";
        continue;
      }
    }
    graph_ops.push_back(op);

    for (auto& var : cfg.Def(op)) {
      // VLOG(3) << "start var " << DebugString(var);
      if (var_can_reused(var) && subblock_output_vars.count(var) == 0) {
        // global op not reuse subblock output vars
        int node_idx_in_pool = -1;
        ir::Node* cached_var = nullptr;
        // VLOG(3) << "match var " << DebugString(var);
        // VLOG(3) << "result " << NodeMatch(var, &cached_var,
        // &node_idx_in_pool);
        if (NodeMatch(var, &cached_var, &node_idx_in_pool)) {
          VLOG(3) << string::Sprintf(
                                     "cache idx %d, pool size %d, var is %s, cached "
                                     "var %s",
                                     node_idx_in_pool, static_cast<int>(pool.size()), DebugString(var),
                                     DebugString(cached_var));
          // VLOG(3) << "matched " << DebugString(var);
          if (var->Name() == cached_var->Name()) {
            PADDLE_THROW(string::Sprintf("Same var!!! %s", DebugString(var)));
          }
          // VLOG(3) << "update graph";
          node_pair_map[op->Op()->GetOrderId()] = std::make_pair(var->Name(), cached_var->Name());
          // VLOG(3) << "reused " << var->Name() << " cache " << cached_var->Name();
          // pool.erase(cached_var);
          pool.Erase(cached_var);
          cfg.UpdateGraph(var, cached_var, idx);
        }
        // VLOG(3) << "match finish";
      }
    }

    for (auto var : cfg.LiveIn(op)) {
      // VLOG(3) << " var " << var->Name() << " can be reuse" << var_can_reused(var) << " live out" << cfg.LiveOut(op).count(var) << " pool" << pool.count(var);
      // if(pool.count(var)) {
      //   VLOG(3) << "pool" << PrintIt(this, pool);
      // }
      if (var_can_reused(var) && cfg.LiveOut(op).count(var) == 0) {
        VLOG(3) << "var " << var->Name() << "inserted";
        VLOG(3) << PrintIt(this, pool);
        // pool.insert(std::make_pair(var, op));
        // pool.emplace_back(it, std::make_pair(var, op));
        pool.Insert(var, op);
        // std::sort(pool.begin(), pool.end(), VarInBytesComparator());
        VLOG(3) << PrintIt(this, pool);
      }
    }
  }
  // fill pool with op order_id and name.
  for (auto& pair : pool) {
    g_pool.push_back(std::make_pair(pair.first->Name(), pair.second->Op()->GetOrderId()));
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
