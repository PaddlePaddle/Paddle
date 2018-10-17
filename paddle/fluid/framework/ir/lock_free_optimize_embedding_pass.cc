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

#include "paddle/fluid/framework/ir/lock_free_optimize_embedding_pass.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

const char LockFreeOptimizeEmbeddingPass::kGradSumOpType[] = "sum";
// TODO(minqiyang): only support sgd at current time, please add
// other optimizers later.
const char LockFreeOptimizeEmbeddingPass::kOptimizerType[] = "sgd";
const char LockFreeOptimizeEmbeddingPass::kEmbeddingGradOpType[] =
    "lookup_table_grad";
const char LockFreeOptimizeEmbeddingPass::kEmbeddingOpType[] = "lookup_table";

std::unique_ptr<ir::Graph> LockFreeOptimizeEmbeddingPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  PADDLE_ENFORCE(graph.get());

  for (auto* node : graph->Nodes()) {
    // Find the sum op after embedding lookup_table_grad op
    if (node->NodeType() == Node::Type::kOperation) {
      if (node->Name() == "lookup_table_grad") {
        LOG(ERROR) << "Found lookup_table_grad op: " << node->Name() << "_"
                   << node->id();
        for (Node* grad_var : node->outputs) {
          LOG(ERROR) << "Found lookup_table_grad op outputs: "
                     << grad_var->Name() << "_" << grad_var->id();
          for (Node* sum_op : grad_var->outputs) {
            LOG(ERROR) << "Found grad_var connects op: " << sum_op->Name()
                       << "_" << sum_op->id();
            for (Node* emb_grad_var : sum_op->outputs) {
              LOG(ERROR) << "Found sum op outputs: " << emb_grad_var->Name()
                         << "_" << emb_grad_var->id();
              // Find the sgd op via sum op
              for (Node* optimize_op : emb_grad_var->inputs) {
                LOG(ERROR) << "Found emb_grad_var upstream op: "
                           << optimize_op->Name() << "_" << optimize_op->id();
              }

              for (Node* optimize_op : emb_grad_var->outputs) {
                LOG(ERROR) << "Found emb_grad_var downstream op: "
                           << optimize_op->Name() << "_" << optimize_op->id();
                for (Node* x : optimize_op->inputs) {
                  LOG(ERROR) << "SGD inputs x: " << x->Name() << "_" << x->id();
                  for (Node* y : x->inputs) {
                    LOG(ERROR) << "x inputs: " << y->Name() << "_" << y->id();
                  }
                }
                for (Node* x : optimize_op->outputs) {
                  LOG(ERROR) << "SGD outputs: " << x->Name() << "_" << x->id();
                }
              }
            }
          }
        }
      }
    }
  }

  std::unordered_set<ir::Node*> sum_op_set;
  for (auto* node : graph->Nodes()) {
    // Find the sum op after embedding lookup_table_grad op
    if (node->NodeType() == Node::Type::kOperation &&
        node->Name() == kEmbeddingGradOpType) {
      // Create new opitimizer node to replace the old one
      ir::Node* new_optimizer_node = nullptr;
      for (Node* lookup_table_grad_output : node->outputs) {
        for (Node* x : lookup_table_grad_output->outputs) {
          LOG(ERROR) << "Before! lookup_table_grad_outputs "
                     << lookup_table_grad_output->Name()
                     << " has: " << x->Name();
        }

        if (lookup_table_grad_output->NodeType() == Node::Type::kVariable &&
            !ir::IsControlDepVar(*lookup_table_grad_output)) {
          LOG(ERROR) << "Start CreateNewOptimizerNode";
          new_optimizer_node = CreateNewOptimizerNode(
              graph.get(), node, lookup_table_grad_output, kOptimizerType,
              lookup_table_grad_output->Name());
          LOG(ERROR) << "End CreateNewOptimizerNode";
          break;
        }
      }

      LOG(ERROR) << "new_optimizer_node: " << new_optimizer_node->Name();

      PADDLE_ENFORCE(new_optimizer_node);

      for (Node* lookup_table_grad_output : node->outputs) {
        auto& next_op_vec = lookup_table_grad_output->outputs;
        LOG(ERROR) << "next_op_vec size: " << next_op_vec.size();
        // redirect lookup_table_grad_output to from sum op to sgd optimizers
        for (auto grad_connected_op_iter = next_op_vec.begin();
             grad_connected_op_iter != next_op_vec.end();) {
          if ((*grad_connected_op_iter)->NodeType() == Node::Type::kOperation) {
            if ((*grad_connected_op_iter)->Name() == kGradSumOpType &&
                !ir::IsControlDepVar(*lookup_table_grad_output)) {
              LOG(ERROR) << "insert to sum_op_set: "
                         << (*grad_connected_op_iter)->Name();
              // store sum op to sum_op_set
              sum_op_set.insert(*grad_connected_op_iter);

              LOG(ERROR) << "remove link: "
                         << (*grad_connected_op_iter)->Name();
              // remove the output link to sum_op
              grad_connected_op_iter =
                  next_op_vec.erase(grad_connected_op_iter);
            } else if ((*grad_connected_op_iter)->Name() == kOptimizerType &&
                       ir::IsControlDepVar(*lookup_table_grad_output)) {
              LOG(ERROR) << "remove link: "
                         << (*grad_connected_op_iter)->Name();
              // remove the output link to optimize_op
              grad_connected_op_iter =
                  next_op_vec.erase(grad_connected_op_iter);
            } else {
              ++grad_connected_op_iter;
            }
          } else {
            ++grad_connected_op_iter;
          }
        }

        // add the output link to sgd optimizers
        next_op_vec.emplace_back(new_optimizer_node);
        new_optimizer_node->inputs.push_back(lookup_table_grad_output);

        for (Node* x : lookup_table_grad_output->outputs) {
          LOG(ERROR) << "Now! lookup_table_grad_outputs "
                     << lookup_table_grad_output->Name()
                     << " has: " << x->Name();
        }
      }
    }
  }

  // Remove the sum_op and its' outputs and connected Optimizers
  for (Node* sum_op : sum_op_set) {
    for (Node* sum_op_output : sum_op->outputs) {
      for (Node* optimize_op : sum_op_output->outputs) {
        if (optimize_op->NodeType() == Node::Type::kOperation &&
            optimize_op->Name() == kOptimizerType) {
          LOG(ERROR) << "remove optimize_op: " << optimize_op->Name() << "_"
                     << optimize_op->id();
          graph->RemoveNode(optimize_op);
        }
      }
      LOG(ERROR) << "remove sum_op_output: " << sum_op_output->Name() << "_"
                 << sum_op_output->id();
      graph->RemoveNode(sum_op_output);
    }
    LOG(ERROR) << "remove sum_op: " << sum_op->Name() << "_" << sum_op->id();
    graph->RemoveNode(sum_op);
  }

  for (auto* node : graph->Nodes()) {
    // Find the sum op after embedding lookup_table_grad op
    if (node->NodeType() == Node::Type::kOperation) {
      if (node->Name() == "lookup_table_grad") {
        LOG(ERROR) << "Found lookup_table_grad op: " << node->Name() << "_"
                   << node->id();
        for (Node* grad_var : node->outputs) {
          LOG(ERROR) << "Found lookup_table_grad op outputs: "
                     << grad_var->Name() << "_" << grad_var->id();
          for (Node* sum_op : grad_var->outputs) {
            LOG(ERROR) << "Found grad_var connects op: " << sum_op->Name()
                       << "_" << sum_op->id();
            // TODO(minqiyang): only support sgd at current time
            for (Node* x : sum_op->inputs) {
              LOG(ERROR) << "SGD inputs x: " << x->Name() << "_" << x->id();
              for (Node* y : x->inputs) {
                LOG(ERROR) << "x inputs: " << y->Name() << "_" << y->id();
              }
            }
            for (Node* x : sum_op->outputs) {
              LOG(ERROR) << "SGD outputs: " << x->Name() << "_" << x->id();
            }
          }
        }
      }
    }
  }

  for (auto* node : graph->Nodes()) {
    for (Node* output_node : node->outputs) {
      if (output_node->Name() == "sgd") {
        LOG(ERROR) << "Output link: " << node->Name() << "_" << node->id()
                   << " --> " << output_node->Name() << "_"
                   << output_node->id();
      }
    }
  }

  // for (Node* input_node : node->inputs) {
  // LOG(ERROR) << "Input link: " << input_node->Name() << "_"
  // << input_node->id() << " --> " << node->Name() << "_"
  // << node->id();
  // }
  // }

  return graph;
}

ir::Node* LockFreeOptimizeEmbeddingPass::CreateNewOptimizerNode(
    ir::Graph* graph, ir::Node* lookup_table_grad,
    ir::Node* lookup_table_grad_output, const std::string& optimizer_type,
    const std::string& grad_name) const {
  for (Node* grad_connected_op : lookup_table_grad_output->outputs) {
    if (grad_connected_op->NodeType() == Node::Type::kOperation &&
        grad_connected_op->Name() == kGradSumOpType) {
      LOG(ERROR) << "Get sum_op node";
      for (Node* sum_op_output : grad_connected_op->outputs) {
        for (Node* optimize_op : sum_op_output->outputs) {
          if (optimize_op->NodeType() == Node::Type::kOperation &&
              optimize_op->Name() == optimizer_type) {
            OpDesc* old_desc = optimize_op->Op();
            // Keep with the same block between new optimizer and the old
            // one
            OpDesc new_desc(*old_desc, old_desc->Block());
            new_desc.SetInput("Param", old_desc->Input("Param"));
            new_desc.SetInput("LearningRate", old_desc->Input("LearningRate"));
            new_desc.SetInput("Grad", std::vector<std::string>({grad_name}));
            new_desc.SetOutput("ParamOut", old_desc->Output("ParamOut"));
            std::vector<std::string> op_role_vars =
                boost::get<std::vector<std::string>>(new_desc.GetAttr(
                    framework::OpProtoAndCheckerMaker::OpRoleVarAttrName()));
            op_role_vars.pop_back();
            op_role_vars.push_back(lookup_table_grad_output->Name());
            for (std::string x : op_role_vars) {
              LOG(ERROR) << "set sgd attrs " << x;
            }
            new_desc.SetAttr(
                framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                op_role_vars);
            new_desc.SetType(optimizer_type);

            new_desc.SetAttr(
                framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                op_role_vars);

            // set backward op's op role var
            lookup_table_grad->Op()->SetAttr(
                framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                op_role_vars);

            // Keep with the same output nodes between new optimizer and the
            // old one
            Node* sgd_node = graph->CreateOpNode(&new_desc);
            LOG(ERROR) << "create new opt node" << sgd_node->Name() << "_"
                       << sgd_node->id();

            for (Node* downstream_node : optimize_op->outputs) {
              ReplaceDownstreamOptimizerNode(downstream_node, optimize_op,
                                             sgd_node);
            }

            for (Node* upstream_node : optimize_op->inputs) {
              // ctrl dependency var node from lookup_table op
              // should be replaced
              if (ir::IsControlDepVar(*upstream_node)) {
                if (IsRelatedEmbeddingOp(upstream_node,
                                         lookup_table_grad_output)) {
                  LOG(ERROR) << "replace related now" << upstream_node->Name();
                  ReplaceUpstreamOptimizerNode(upstream_node, optimize_op,
                                               sgd_node);
                }
              }

              // SGD must have only one param and LR in
              PADDLE_ENFORCE(old_desc->Input("LearningRate").size() == 1u);
              PADDLE_ENFORCE(old_desc->Input("Param").size() == 1u);

              // LR and weight nodes should be copied
              if (upstream_node->Name() == old_desc->Input("LearningRate")[0] ||
                  upstream_node->Name() == old_desc->Input("Param")[0]) {
                LOG(ERROR) << "replace now" << upstream_node->Name();
                ReplaceUpstreamOptimizerNode(upstream_node, optimize_op,
                                             sgd_node);
              }
            }

            return sgd_node;
          }
        }
      }
    }
  }

  return nullptr;
}

void LockFreeOptimizeEmbeddingPass::ReplaceUpstreamOptimizerNode(
    ir::Node* upstream_node, ir::Node* old_optimizer_node,
    ir::Node* new_optimizer_node) const {
  PADDLE_ENFORCE(upstream_node);
  PADDLE_ENFORCE(old_optimizer_node);
  PADDLE_ENFORCE(new_optimizer_node);

  // Remove the old_optimizer_node from upstream_node's outputs vector
  auto& output_node_vec = upstream_node->outputs;
  for (auto output_node_iter = output_node_vec.begin();
       output_node_iter != output_node_vec.end();) {
    if (*output_node_iter == old_optimizer_node) {
      LOG(ERROR) << "Found old_optimizer_node and remove it "
                 << (*output_node_iter)->Name();
      LOG(ERROR) << "From upstream node " << upstream_node->Name();
      output_node_vec.erase(output_node_iter);
      break;
    } else {
      ++output_node_iter;
    }
  }

  // Add the new_optimizer_node to upstream_node's outputs vector
  output_node_vec.emplace_back(new_optimizer_node);
  new_optimizer_node->inputs.emplace_back(upstream_node);
}

void LockFreeOptimizeEmbeddingPass::ReplaceDownstreamOptimizerNode(
    ir::Node* downstream_node, ir::Node* old_optimizer_node,
    ir::Node* new_optimizer_node) const {
  PADDLE_ENFORCE(downstream_node);
  PADDLE_ENFORCE(old_optimizer_node);
  PADDLE_ENFORCE(new_optimizer_node);

  // Remove the old_optimizer_node from downstream_node's inputs vector
  auto& input_node_vec = downstream_node->inputs;
  for (auto input_node_iter = input_node_vec.begin();
       input_node_iter != input_node_vec.end();) {
    if (*input_node_iter == old_optimizer_node) {
      LOG(ERROR) << "Found old_optimizer_node and remove it "
                 << (*input_node_iter)->Name();
      input_node_vec.erase(input_node_iter);
      break;
    } else {
      ++input_node_iter;
    }
  }

  // Add the new_optimizer_node to downstream_node's inputs vector
  input_node_vec.emplace_back(new_optimizer_node);
  new_optimizer_node->outputs.emplace_back(downstream_node);
}

bool LockFreeOptimizeEmbeddingPass::IsRelatedEmbeddingOp(
    ir::Node* ctrl_dep_var_node,
    ir::Node* lookup_table_grad_output_node) const {
  PADDLE_ENFORCE(ir::IsControlDepVar(*ctrl_dep_var_node));

  if (ctrl_dep_var_node->inputs.size() == 1u &&
      ctrl_dep_var_node->inputs[0]->Name() == kEmbeddingOpType) {
    for (Node* lookup_table_grad : lookup_table_grad_output_node->inputs) {
      if (lookup_table_grad->Name() == kEmbeddingGradOpType) {
        Node* lookup_table_node = ctrl_dep_var_node->inputs[0];
        LOG(ERROR) << "found lookup_table_node " << lookup_table_node->Name()
                   << "_" << lookup_table_node->id();
        Node* grad_ids_input_node = nullptr;
        for (auto& ids_param_name : lookup_table_grad->Op()->Input("Ids")) {
          for (Node* lookup_table_grad_input : lookup_table_grad->inputs) {
            if (lookup_table_grad_input->Name() == ids_param_name) {
              grad_ids_input_node = lookup_table_grad_input;
              LOG(ERROR) << "found grad_ids_input_node "
                         << grad_ids_input_node->Name();
              break;
            }
          }
        }
        PADDLE_ENFORCE(grad_ids_input_node);

        Node* ids_input_node = nullptr;
        for (auto& ids_param_name : lookup_table_node->Op()->Input("Ids")) {
          for (Node* lookup_table_input : lookup_table_node->inputs) {
            if (lookup_table_input->Name() == ids_param_name) {
              ids_input_node = lookup_table_input;
              LOG(ERROR) << "found ids_input_node " << ids_input_node->Name();
              break;
            }
          }
        }
        PADDLE_ENFORCE(ids_input_node);

        if (ids_input_node == grad_ids_input_node) {
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
    }

    // if hit here, then embedding op is NOT found, so we return false
    return false;
  }

  return false;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(lock_free_optimize_embedding_pass,
              paddle::framework::ir::LockFreeOptimizeEmbeddingPass);
