/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/constant_folding_pass.h"
#include <string>
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

#include "paddle/fluid/framework/convert_utils.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

/*
 * When a op's inputs and outputs is determined before feeding data to the
 * model, we can remove this op from the model. This ConstantFolding pass can
 * remove all these like ops.
 *
 */

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct ConstantFolding : public PatternBase {
  ConstantFolding(PDPattern* pattern, const std::string& name_scope)
      : PatternBase(pattern, name_scope, "constant_folding_pass") {}
};
}  // namespace patterns

static void GetUniqueVarsNode(
    Graph* graph,
    std::unordered_map<std::string, Node*>* vars,
    std::vector<Graph*>* sub_graphs,
    std::vector<std::unordered_set<std::string>>* all_output_var_names,
    std::unordered_map<int, Node*>* control_flow_ops,
    std::unordered_set<std::string>* forbidden_folding_vars) {
  auto graph_size = graph->SubGraphsSize();
  std::vector<std::vector<Node*>> all_op_nodes;
  sub_graphs->resize(graph_size);
  all_op_nodes.resize(graph_size);
  all_output_var_names->resize(graph_size);
  VLOG(4) << "graph size: " << graph_size;
  for (size_t i = 0; i < graph_size; i++) {
    auto sub_graph = graph->GetSubGraph(i);
    (*sub_graphs)[i] = sub_graph;
    all_op_nodes[i] = TopologySortOperations(*sub_graph);
    for (auto* var_node : sub_graph->Nodes()) {
      if (!var_node->IsVar()) continue;
      auto var_name = var_node->Var()->Name();
      if (vars->count(var_name) == 0) {
        (*vars)[var_name] = var_node;
        VLOG(4) << var_name << " is in graph " << i;
      }
    }
  }

  for (int i = 0; i < graph_size; i++) {
    for (auto* op_node : all_op_nodes[i]) {
      if (op_node->Op()->HasAttr("sub_block")) {
        auto* sub_block = PADDLE_GET_CONST(framework::BlockDesc*,
                                           op_node->Op()->GetAttr("sub_block"));
        (*control_flow_ops)[sub_block->ID()] = op_node;
        std::unordered_set<std::string> set;
        for (auto in_var : op_node->inputs) {
          set.insert(in_var->Name());
        }
        for (auto out_var : op_node->outputs) {
          if (set.find(out_var->Name()) != set.end()) {
            (*forbidden_folding_vars).insert(out_var->Name());
          }
        }
      }
      for (auto* var_out_node : op_node->outputs) {
        if (var_out_node->IsVar()) {
          (*all_output_var_names)[i].insert(var_out_node->Var()->Name());
        }
      }
    }
  }
}

ConstantFoldingPass::ConstantFoldingPass() {}

void DoConstantFolding(
    Scope* scope,
    const std::vector<Graph*>& graphs,
    const std::unordered_map<std::string, Node*>& unique_vars_node,
    const std::vector<std::unordered_set<std::string>>& all_output_var_names,
    const std::unordered_map<int, Node*>& control_flow_ops,
    const std::unordered_set<std::string>& forbidden_folding_vars,
    int parent_graph_index,
    int subgraph_index) {
  std::vector<std::string> blacklist{"feed", "matrix_multiply", "save"};
  auto op_sorted_nodes = TopologySortOperations(*graphs[subgraph_index]);
  for (auto* op_node : op_sorted_nodes) {
    if (!op_node->IsOp()) continue;
    if (std::find(blacklist.begin(), blacklist.end(), op_node->Name()) !=
        blacklist.end())
      continue;
    if (op_node->Op()->HasAttr("sub_block")) {
      auto* sub_block = PADDLE_GET_CONST(framework::BlockDesc*,
                                         op_node->Op()->GetAttr("sub_block"));
      DoConstantFolding(scope,
                        graphs,
                        unique_vars_node,
                        all_output_var_names,
                        control_flow_ops,
                        forbidden_folding_vars,
                        graphs[subgraph_index]->GetBlockId(),
                        sub_block->ID());
    }
    bool input_persis = true;
    // map is used to record how many time a name string occurs in the whole
    // graph's nodes
    std::unordered_map<std::string, int> map;
    for (auto in_node : op_node->inputs) {
      map[in_node->Name()] = 0;
      auto real_unique_in_node = unique_vars_node.at(in_node->Name());
      if (!real_unique_in_node->Var()->Persistable()) {
        input_persis = false;
        break;
      }
    }
    for (auto out_node : op_node->outputs) {
      map[out_node->Name()] = 0;
    }
    // Forbid other node in graph having the same name with nodes in map
    for (auto iter : map) {
      for (int i = 0; i < all_output_var_names.size(); i++) {
        for (auto var_name : all_output_var_names[i]) {
          if (var_name == iter.first) {
            // If a tensor of the input and output of the op is both the input
            // and output of the while, it cannot be folded.
            if (forbidden_folding_vars.find(var_name) !=
                forbidden_folding_vars.end()) {
              input_persis = false;
              break;
            }
            map[var_name]++;
          }
        }
      }
      // If the output variables in the subgraph are only the output of while
      // op, and the operators in the subgraph can be collapsed, then this
      // situation should be collapsed and should not be limited by the number
      // of output names in while op. Exp: while -> while -> while ->
      // fill_constant
      if (parent_graph_index != subgraph_index) {
        int son_control_flow_op_block_index = subgraph_index;
        while (control_flow_ops.count(son_control_flow_op_block_index)) {
          auto* parent_control_flow_op_node =
              control_flow_ops.at(son_control_flow_op_block_index);
          for (auto* out_var_node : parent_control_flow_op_node->outputs) {
            if (out_var_node->Var()->Name() == iter.first) {
              map[iter.first]--;
            }
            son_control_flow_op_block_index = out_var_node->GetVarNodeBlockId();
          }
        }
        if (map[iter.first] > 1) {
          input_persis = false;
        }
      }
    }

    framework::Scope* local_scope = new framework::Scope();
    std::unordered_set<const paddle::framework::ir::Node*>
        sub_graph_remove_nodes;
    std::unique_ptr<OperatorBase> op;

    if (input_persis) {
      for (auto in_node : op_node->inputs) {
        local_scope->Var(in_node->Var()->Name());
        local_scope->FindVar(in_node->Var()->Name())
            ->GetMutable<phi::DenseTensor>();
        // This persistable input node is exclusive, and can be removed
        if (in_node->outputs.size() == 1L)
          sub_graph_remove_nodes.emplace(in_node);
        auto* unique_in_node = unique_vars_node.at(in_node->Name());

        auto in_shape = in_node->Var()->GetShape();
        auto* global_persis_x_tensor =
            scope->FindVar(in_node->Name())->GetMutable<phi::DenseTensor>();
        auto* local_x_tensor = local_scope->FindVar(in_node->Name())
                                   ->GetMutable<phi::DenseTensor>();
        local_x_tensor->Resize(global_persis_x_tensor->dims());
        *local_x_tensor = *global_persis_x_tensor;
      }

      op = paddle::framework::OpRegistry::CreateOp(*op_node->Op());
      sub_graph_remove_nodes.emplace(op_node);
      for (auto out_node : op_node->outputs) {
        local_scope->Var(out_node->Var()->Name());
        local_scope->FindVar(out_node->Var()->Name())
            ->GetMutable<phi::DenseTensor>();
        // useless out_node can be removed, not need set it persistable !
        if (out_node->outputs.size() == 0L)
          sub_graph_remove_nodes.emplace(out_node);
      }
      op->Run(*local_scope, platform::CPUPlace());
      for (auto out_node : op_node->outputs) {
        // this out_node is useless, do not set it persistable
        auto* local_out_tensor = local_scope->FindVar(out_node->Name())
                                     ->GetMutable<phi::DenseTensor>();
        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < local_out_tensor->dims().size(); i++) {
          out_shape.push_back(local_out_tensor->dims()[i]);
        }
        auto real_unique_out_node = unique_vars_node.at(out_node->Name());
        // Although variables in different subgraphs have the same name,
        // but they correspond to different nodes. So we can remove the subgraph
        // node, but need to synchronize with other graphs.
        if (real_unique_out_node->outputs.size() != 0) {
          auto out_desc = real_unique_out_node->Var();
          out_desc->SetShape(out_shape);
          out_desc->SetPersistable(true);
          auto* global_out_tensor =
              scope->Var(out_node->Name())->GetMutable<phi::DenseTensor>();
          *global_out_tensor = *local_out_tensor;
        }
        if (out_node->outputs.size() != 0L) {
          auto* var_desc_out = op_node->Op()->Block()->Var(out_node->Name());
          var_desc_out->SetShape(out_shape);
          var_desc_out->SetPersistable(true);
          var_desc_out->Flush();
        }
      }
      GraphSafeRemoveNodes(graphs[subgraph_index], sub_graph_remove_nodes);
    }
    delete local_scope;
  }
}

void ConstantFoldingPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("constant_folding", graph);
  auto* scope = param_scope();

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "scope must not be null when applying constant folding."));

  // var name -> real var node
  std::unordered_map<std::string, Node*> unique_vars_node;
  std::vector<Graph*> graphs;
  std::vector<std::unordered_set<std::string>> all_output_var_names;
  std::unordered_set<std::string> forbidden_folding_vars;
  std::unordered_map<int, Node*> control_flow_ops;
  GetUniqueVarsNode(graph,
                    &unique_vars_node,
                    &graphs,
                    &all_output_var_names,
                    &control_flow_ops,
                    &forbidden_folding_vars);
  DoConstantFolding(scope,
                    graphs,
                    unique_vars_node,
                    all_output_var_names,
                    control_flow_ops,
                    forbidden_folding_vars,
                    0,
                    0);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(constant_folding_pass,
              paddle::framework::ir::ConstantFoldingPass);
