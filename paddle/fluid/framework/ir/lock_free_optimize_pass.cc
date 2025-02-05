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

#include "paddle/fluid/framework/ir/lock_free_optimize_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

const char kSumGradOpName[] = "sum";  // NOLINT
// TODO(minqiyang): only support sgd at current time, please add
// other optimizers later.
const char kOptimizerType[] = "sgd";  // NOLINT

void LockFreeOptimizePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, common::errors::InvalidArgument("Graph cannot be nullptr."));

  // We could collect all weights' name from SGD, where
  // W1 <- SGD(W0, Grad0)
  std::unordered_set<std::string> weight_var_set;
  for (auto* node : graph->Nodes()) {
    if (IsOpNamed(node, kOptimizerType)) {
      auto& param_out_vars = node->Op()->Output("ParamOut");
      PADDLE_ENFORCE_EQ(
          param_out_vars.size(),
          1u,
          common::errors::InvalidArgument(
              "In op(%s), find output(ParamOut) failed.", node->Name()));
      weight_var_set.insert(param_out_vars[0]);
    }
  }

  // find all grad's merge op via weight name, where
  // Grad0 <- SUM(Grad1, Grad2, Grad3 ...)
  std::unordered_set<ir::Node*> grad_sum_op_set;
  for (ir::Node* node : graph->Nodes()) {
    if (IsOpNamed(node, kSumGradOpName)) {
      for (ir::Node* output : node->outputs) {
        // strip the last grad suffix @GRAD
        std::string var_name = output->Name();
        const std::string suffix(kGradVarSuffix);
        if (var_name != suffix && var_name.size() > suffix.size() &&
            var_name.substr(var_name.size() - suffix.size()) == suffix) {
          // if so then strip them off
          var_name = var_name.substr(0, var_name.size() - suffix.size());
          if (weight_var_set.find(var_name) != weight_var_set.end()) {
            grad_sum_op_set.insert(node);
            break;
          }
        }
      }
    }
  }

  // get the forward op and backward op pairs, where
  // out <- forward(X, W)
  // Grad1 <- backward(out, X')
  // Grad0 <- SUM(Grad1, Grad2, Grad3 ...)
  // W0 <- SGD(W1, Grad0)
  for (ir::Node* node : grad_sum_op_set) {
    for (ir::Node* merged_grad_var : node->outputs) {
      // find the optimizers connected with sum op
      if (IsVarNameEndsWith(merged_grad_var, kGradVarSuffix) &&
          merged_grad_var->outputs.size() == 1u) {
        ir::Node* opt_node = merged_grad_var->outputs[0];
        VLOG(3) << "Found opt node " << opt_node->Name();

        // find the backward op connected with sum op
        for (ir::Node* unmerged_grad_var : node->inputs) {
          if (IsVarNameContains(unmerged_grad_var, kGradVarSuffix) &&
              unmerged_grad_var->inputs.size() == 1u) {
            ir::Node* backward_op = unmerged_grad_var->inputs[0];

            VLOG(3) << "Found backward_op " << backward_op->Name();

            // find the forward op related to the backward op
            ir::Node* forward_op =
                FindForwardOpViaBackwardOp(graph, backward_op);

            VLOG(3) << "Found forward_op " << forward_op->Name();

            PADDLE_ENFORCE_NOT_NULL(
                forward_op,
                common::errors::NotFound(
                    "Can not find forward op for backward op(%s).",
                    backward_op->Name()));

            Node* new_optimizer_node = CreateNewSGDNode(
                graph, forward_op, backward_op, node, opt_node);

            PADDLE_ENFORCE_NOT_NULL(
                new_optimizer_node,
                common::errors::InvalidArgument(
                    "Create new SGD node failed, backward op is %s.",
                    backward_op->Name()));
          }
        }
      }
    }
  }

  // Remove the sum_op and its' outputs and connected Optimizers
  for (Node* sum_op : grad_sum_op_set) {
    for (Node* sum_op_output : sum_op->outputs) {
      for (Node* optimize_op : sum_op_output->outputs) {
        if (optimize_op->NodeType() == Node::Type::kOperation &&
            optimize_op->Name() == kOptimizerType) {
          VLOG(3) << "remove optimize_op: " << optimize_op->Name() << "_"
                  << optimize_op->id();
          graph->RemoveNode(optimize_op);
        }
      }
      VLOG(3) << "remove sum_op_output: " << sum_op_output->Name() << "_"
              << sum_op_output->id();
      graph->RemoveNode(sum_op_output);
    }
    VLOG(3) << "remove sum_op: " << sum_op->Name() << "_" << sum_op->id();
    graph->RemoveNode(sum_op);
  }

  for (auto* node : graph->Nodes()) {
    for (Node* output_node : node->outputs) {
      if (output_node->Name() == "sgd") {
        VLOG(3) << "Node link to SGD: " << node->Name() << "_" << node->id()
                << " --> " << output_node->Name() << "_" << output_node->id();
        for (Node* input_node : node->inputs) {
          VLOG(3) << "SGD Input link: " << input_node->Name() << "_"
                  << input_node->id() << " --> " << node->Name() << "_"
                  << node->id();
        }
      }
    }
  }
}

ir::Node* LockFreeOptimizePass::CreateNewSGDNode(
    ir::Graph* graph,
    ir::Node* forward_node,
    ir::Node* backward_node,
    ir::Node* grad_sum_node,
    ir::Node* optimize_node) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Input argument graph cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      forward_node,
      common::errors::InvalidArgument(
          "Input argument forward_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      backward_node,
      common::errors::InvalidArgument(
          "Input argument backward_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      grad_sum_node,
      common::errors::InvalidArgument(
          "Input argument grad_sum_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      optimize_node,
      common::errors::InvalidArgument(
          "Input argument optimize_node cannot be nullptr."));

  // find the grad var node between the grad sum node and backward_node
  std::vector<ir::Node*> grad_vars =
      FindConnectedNode(backward_node, grad_sum_node);
  ir::Node* grad_node = nullptr;
  for (ir::Node* node : grad_vars) {
    if (!ir::IsControlDepVar(*node)) {
      grad_node = node;
    }
  }
  PADDLE_ENFORCE_NOT_NULL(
      grad_node,
      common::errors::NotFound("Can not find control dep variable."));

  // create a new SGD node
  OpDesc* old_desc = optimize_node->Op();
  // keep with the same block between new optimizer and the old one
  OpDesc new_desc(*old_desc, old_desc->Block());
  new_desc.SetInput("Param", old_desc->Input("Param"));
  new_desc.SetInput("LearningRate", old_desc->Input("LearningRate"));
  new_desc.SetInput("Grad", std::vector<std::string>({grad_node->Name()}));
  new_desc.SetOutput("ParamOut", old_desc->Output("ParamOut"));

  std::vector<std::string> op_role_vars = PADDLE_GET_CONST(
      std::vector<std::string>,
      new_desc.GetAttr(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName()));
  // replace the second op role var, because the grad name was
  // changed in new optimizer
  op_role_vars.pop_back();
  op_role_vars.push_back(grad_node->Name());
  new_desc.SetAttr(framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(),
                   op_role_vars);
  new_desc.SetType(kOptimizerType);

  // set backward op's op role var, this will be used to
  // set device_id in multi_device_pass
  backward_node->Op()->SetAttr(
      framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(), op_role_vars);
  // backward_node->Op()->SetAttr(
  // framework::OpProtoAndCheckerMaker::OpRoleVarAttrName(), {});

  // keep with the same output nodes between new optimizer and the
  // old one
  Node* sgd_node = graph->CreateOpNode(&new_desc);

  // change all outputs of the optimize_node to the new one
  ReplaceAllDownstreamNode(optimize_node, sgd_node);

  // find connected node between forward node and optimize node
  // and replace the optimize node to new sgd node
  std::vector<ir::Node*> forward_opt_connected_nodes =
      FindConnectedNode(forward_node, optimize_node);
  for (ir::Node* node : forward_opt_connected_nodes) {
    ReplaceUpstreamNode(node, optimize_node, sgd_node);
  }

  // find connected node between backward node and optimize node
  // and replace the optimize node to new sgd node
  std::vector<ir::Node*> backward_opt_connected_nodes =
      FindConnectedNode(backward_node, optimize_node);
  for (ir::Node* node : backward_opt_connected_nodes) {
    ReplaceUpstreamNode(node, optimize_node, sgd_node);
  }

  // SGD must have only one param and LR in
  PADDLE_ENFORCE_EQ(
      old_desc->Input("LearningRate").size(),
      1u,
      common::errors::InvalidArgument(
          "In op(%s), find input(LearningRate) failed.", old_desc->Type()));
  PADDLE_ENFORCE_EQ(
      old_desc->Input("Param").size(),
      1u,
      common::errors::InvalidArgument("In op(%s), find input(Param) failed.",
                                      old_desc->Type()));

  // LR and weight nodes should be copied
  for (Node* upstream_node : optimize_node->inputs) {
    if (upstream_node->Name() == old_desc->Input("LearningRate")[0] ||
        upstream_node->Name() == old_desc->Input("Param")[0]) {
      ReplaceUpstreamNode(upstream_node, optimize_node, sgd_node);
    }
  }

  VLOG(3) << "Create new opt node" << sgd_node->Name() << "_" << sgd_node->id();

  return sgd_node;
}

std::vector<ir::Node*> LockFreeOptimizePass::FindConnectedNode(
    ir::Node* upstream_node, ir::Node* downstream_node) const {
  std::vector<ir::Node*> result;
  for (ir::Node* out_node : upstream_node->outputs) {
    for (ir::Node* in_node : downstream_node->inputs) {
      if (in_node == out_node) {
        result.push_back(in_node);
      }
    }
  }

  return result;
}

void LockFreeOptimizePass::ReplaceUpstreamNode(
    ir::Node* upstream_node,
    ir::Node* old_optimizer_node,
    ir::Node* new_optimizer_node) const {
  PADDLE_ENFORCE_NOT_NULL(
      upstream_node,
      common::errors::InvalidArgument(
          "Input argument upstream_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      old_optimizer_node,
      common::errors::InvalidArgument(
          "Input argument old_optimizer_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      new_optimizer_node,
      common::errors::InvalidArgument(
          "Input argument new_optimizer_node cannot be nullptr."));

  // Remove the old_optimizer_node from upstream_node's outputs vector
  auto& output_node_vec = upstream_node->outputs;
  for (auto output_node_iter = output_node_vec.begin();
       output_node_iter != output_node_vec.end();) {
    if (*output_node_iter == old_optimizer_node) {
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

void LockFreeOptimizePass::ReplaceAllDownstreamNode(
    ir::Node* old_optimizer_node, ir::Node* new_optimizer_node) const {
  PADDLE_ENFORCE_NOT_NULL(
      old_optimizer_node,
      common::errors::InvalidArgument(
          "Input argument old_optimizer_node cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      new_optimizer_node,
      common::errors::InvalidArgument(
          "Input argument new_optimizer_node cannot be nullptr."));

  for (ir::Node* downstream_node : old_optimizer_node->outputs) {
    // Remove the old_optimizer_node from downstream_node's inputs vector
    auto& input_node_vec = downstream_node->inputs;
    for (auto input_node_iter = input_node_vec.begin();
         input_node_iter != input_node_vec.end();) {
      if (*input_node_iter == old_optimizer_node) {
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
}

ir::Node* LockFreeOptimizePass::FindForwardOpViaBackwardOp(
    ir::Graph* graph, ir::Node* backward_node) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          common::errors::InvalidArgument(
                              "Input argument graph cannot be nullptr."));
  PADDLE_ENFORCE_NOT_NULL(
      backward_node,
      common::errors::InvalidArgument(
          "Input argument backward_node cannot be nullptr."));

  // strip the suffix _grad of backward_node's name
  std::string forward_op_name = backward_node->Name();
  const std::string suffix("_grad");
  if (forward_op_name != suffix && forward_op_name.size() > suffix.size() &&
      forward_op_name.substr(forward_op_name.size() - suffix.size()) ==
          suffix) {
    // if so then strip them off
    forward_op_name =
        forward_op_name.substr(0, forward_op_name.size() - suffix.size());
  } else {
    LOG(WARNING) << "Illegal backward node's name " << backward_node->Name()
                 << " id " << backward_node->id();

    return nullptr;
  }

  for (ir::Node* node : graph->Nodes()) {
    if (node->Name() == forward_op_name) {
      if (node->outputs.empty()) {
        // if forward_node has no output, then it has NO grad op
        continue;
      }

      // check whether all inputs of the backward_op that ends_with @GRAD
      // comes from the output of forward_op is the input of the backward_op
      bool is_related_forward_node = true;
      for (ir::Node* backward_input : backward_node->inputs) {
        if (IsVarNameEndsWith(backward_input, kGradVarSuffix)) {
          bool meets_correct_output = false;
          for (ir::Node* forward_output : node->outputs) {
            if (forward_output->Name() + kGradVarSuffix ==
                backward_input->Name()) {
              meets_correct_output = true;
              break;
            }
          }

          if (!meets_correct_output) {
            is_related_forward_node = false;
            break;
          }
        }
      }

      if (is_related_forward_node) {
        return node;
      }
    }
  }

  return nullptr;
}

}  // namespace paddle::framework::ir

REGISTER_PASS(lock_free_optimize_pass,
              paddle::framework::ir::LockFreeOptimizePass);
