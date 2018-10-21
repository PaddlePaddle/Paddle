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
#include <iterator>
#include <vector>

namespace paddle {
namespace framework {
namespace details {

using details::UnlivedNodePool;
using details::ReusedNodePairMap;
using details::ControlFlowGraph;

bool AnalysisVarPass::NodeMatching(ir::Node* var, ir::Node** cache,
                                   int* node_idx_in_pool) const {
  auto compare_node_type_and_shape = [&](ir::Node* lhs, ir::Node* rhs) {
    auto* lhs_desc = lhs->Var();
    auto* rhs_desc = rhs->Var();
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();
    if ((lhs_shape[0] == -1) ^ (rhs_shape[0] == -1)) return false;
    if (lhs_desc->GetDataType() != rhs_desc->GetDataType() ||
        lhs_desc->GetType() != rhs_desc->GetType())
      return false;
    auto lhs_size =
        std::abs(std::accumulate(lhs_shape.begin(), lhs_shape.end(), 1));
    auto rhs_size =
        std::abs(std::accumulate(rhs_shape.begin(), rhs_shape.end(), 1));
    return lhs_size == rhs_size;
  };

  for (auto it = pool.begin(); it != pool.end(); ++it) {
    ir::Node* cached_var = *it;
    if (compare_node_type_and_shape(cached_var, var)) {
      *cache = cached_var;
      *node_idx_in_pool = std::distance(it, pool.begin());
      return true;
    }
  }
  return false;
}

const std::string AnalysisVarPass::DebugString(ir::Node* var) const {
  std::stringstream ss;
  ss << var->Name();
  ss << "[";
  auto shape = var->Var()->GetShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != shape.size() - 1) {
      ss << shape[i] << ",";
    } else {
      ss << shape[i];
    }
  }
  ss << "]";
  return ss.str();
}

std::unique_ptr<ir::Graph> AnalysisVarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  // auto& node_pool = Get<UnlivedNodePool>(kGlobalUnlivedNodePool);
  // auto &fetched_vars = Get<std::unordered_set<std::string>>(kFetchedVars);
  auto& node_pair_map = Get<ReusedNodePairMap>(kGlobalReusedNodePairMap);
  auto& graph_ops = Get<std::vector<ir::Node*>>(kGraphReusedOps);

  // use worklist algorithm to compute the unlived variables.
  ControlFlowGraph cfg(*graph.get());
  cfg.LiveVariableAnalysis();

  auto op_has_subblock = [&](const AttributeMap& attrs) {
    for (auto& attr : attrs) {
      if (typeid(attr.second) == typeid(BlockDesc)) return true;
    }
    return false;
  };

  auto var_can_reused = [&](ir::Node* node) {
    PADDLE_ENFORCE(node->IsVar(), string::Sprintf("Expect node %s as Variable.",
                                                  node->Name()));
    auto* desc = node->Var();
    proto::VarType::Type type = desc->GetType();
    // if (fetched_vars.count(node)) {return false;}
    if (desc->Persistable() || (type != proto::VarType::LOD_TENSOR &&
                                type != proto::VarType::SELECTED_ROWS) ||
        desc->GetShape().size() == 0) {
      return false;
    }
    for (auto* op : node->inputs) {
      // fill_constant on cpu can not be shared.
      if (op->Name() == "fill_constant" && op->Op()->HasAttr("force_cpu")) {
        return false;
      }
    }
    return true;
  };

  for (auto& op : cfg.Ops()) {
    auto* op_desc = op->Op();
    if (op_has_subblock(op_desc->GetAttrMap())) continue;
    graph_ops.push_back(op);

    for (auto var : cfg.LiveIn(op)) {
      if (!cfg.LiveOut(op).count(var) && var_can_reused(var)) {
        pool.insert(var);
      }
    }

    for (auto& var : cfg.Def(op)) {
      if (var_can_reused(var)) {
        int node_idx_in_pool = -1;
        ir::Node* cached_var = nullptr;
        if (NodeMatching(var, &cached_var, &node_idx_in_pool)) {
          VLOG(3) << string::Sprintf(
              "Hit Cache !!! cache pool index %d, var is %s, cached var %s",
              node_idx_in_pool, DebugString(var), DebugString(cached_var));
          cfg.UpdateGraph(var, cached_var, node_idx_in_pool);
          pool.erase(cached_var);
          node_pair_map[op] = std::make_pair(var, cached_var);
        }
      }
    }
  }
  return graph;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(analysis_var_pass, paddle::framework::details::AnalysisVarPass)
    .RequirePassAttr(paddle::framework::details::kGlobalReusedNodePairMap)
    .RequirePassAttr(paddle::framework::details::kGraphReusedOps);
