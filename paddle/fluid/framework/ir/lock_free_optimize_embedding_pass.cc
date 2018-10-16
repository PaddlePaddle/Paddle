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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

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
              for (Node* optimize_op : emb_grad_var->outputs) {
                LOG(ERROR) << "Found emb_grad_var connects op: "
                           << optimize_op->Name() << "_" << optimize_op->id();
                // TODO(minqiyang): only support sgd at current time
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
    if (node->NodeType() == Node::Type::kOperation) {
      if (node->Name() == "lookup_table_grad") {
        ir::Node* new_optimizer_node = nullptr;
        for (Node* lookup_table_grad_output : node->outputs) {
          if (lookup_table_grad_output->NodeType() == Node::Type::kVariable &&
              !ir::IsControlDepVar(*lookup_table_grad_output)) {
            // TODO(minqiyang): only support sgd at current time, please add
            // other
            // optimizers later.
            new_optimizer_node = CreateNewOptimizerNode(
                graph.get(), node, "sgd", lookup_table_grad_output->Name());
            break;
          }
        }

        PADDLE_ENFORCE(new_optimizer_node);

        for (Node* lookup_table_grad_output : node->outputs) {
          // redirect lookup_table_grad_output to from sum op to sgd optimizers
          for (auto grad_connected_op_iter =
                   lookup_table_grad_output->outputs.begin();
               grad_connected_op_iter !=
               lookup_table_grad_output->outputs.end();
               ++grad_connected_op_iter) {
            if ((*grad_connected_op_iter)->NodeType() ==
                    Node::Type::kOperation &&
                (*grad_connected_op_iter)->Name() == "sum") {
              // store sum op to sum_op_set
              sum_op_set.insert(*grad_connected_op_iter);

              // remove the output link to sum_op
              lookup_table_grad_output->outputs.erase(grad_connected_op_iter);

              // add the output link to sgd optimizers
              lookup_table_grad_output->outputs.emplace_back(
                  new_optimizer_node);
              new_optimizer_node->inputs.push_back(lookup_table_grad_output);
            }
          }
        }
      }
    }
  }

  // Remove the sum_op and its' outputs and connected Optimizers
  for (Node* sum_op : sum_op_set) {
    for (Node* sum_op_output : sum_op->outputs) {
      for (Node* optimize_op : sum_op_output->outputs) {
        // TODO(minqiyang): only support sgd at current time, please add other
        // optimizers later.
        if (optimize_op->NodeType() == Node::Type::kOperation &&
            optimize_op->Name() == "sgd") {
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

  // for (Node* output_node : node->outputs) {
  // LOG(ERROR) << "Output link: " << node->Name() << "_"
  // << node->id() << " --> " << output_node->Name() << "_"
  // << output_node->id();
  // }

  // for (Node* input_node : node->inputs) {
  // LOG(ERROR) << "Input link: " << input_node->Name() << "_"
  // << input_node->id() << " --> " << node->Name() << "_"
  // << node->id();
  // }
  // }

  return graph;
}

ir::Node* LockFreeOptimizeEmbeddingPass::CreateNewOptimizerNode(
    ir::Graph* graph, ir::Node* node, const std::string& optimizer_type,
    const std::string& grad_name) const {
  for (Node* lookup_table_grad_output : node->outputs) {
    if (lookup_table_grad_output->NodeType() == Node::Type::kVariable) {
      for (Node* grad_connected_op : lookup_table_grad_output->outputs) {
        if (grad_connected_op->NodeType() == Node::Type::kOperation &&
            grad_connected_op->Name() == "sum") {
          for (Node* sum_op_output : grad_connected_op->outputs) {
            for (Node* optimize_op : sum_op_output->outputs) {
              if (optimize_op->NodeType() == Node::Type::kOperation &&
                  optimize_op->Name() == optimizer_type) {
                OpDesc* old_desc = optimize_op->Op();
                // Keep with the same block between new optimizer and the old
                // one
                OpDesc new_desc(*old_desc, old_desc->Block());
                new_desc.SetInput("Param", old_desc->Input("Param"));
                new_desc.SetInput("LearningRate",
                                  old_desc->Input("LearningRate"));
                new_desc.SetInput("Grad",
                                  std::vector<std::string>({grad_name}));
                new_desc.SetOutput("ParamOut", old_desc->Output("ParamOut"));
                new_desc.SetType(optimizer_type);

                // Keep with the same output nodes between new optimizer and the
                // old one
                Node* sgd_node = graph->CreateOpNode(&new_desc);
                sgd_node->outputs.insert(sgd_node->outputs.end(),
                                         optimize_op->outputs.begin(),
                                         optimize_op->outputs.end());
                return sgd_node;
              }
            }
          }
        }
      }
    }
  }

  return nullptr;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(lock_free_optimize_embedding_pass,
              paddle::framework::ir::LockFreeOptimizeEmbeddingPass);
