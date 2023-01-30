// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2023 NVIDIA Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/fuse_adamw_op_pass.h"
#include <string>
#include "paddle/fluid/framework/details/multi_devices_helper.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

void FuseAdamWPass::ApplyImpl(ir::Graph *graph) const {
  graph = FuseAdamWFun(graph, true, true);
  graph = FuseAdamWFun(graph, true, false);
  graph = FuseAdamWFun(graph, false, true);
  graph = FuseAdamWFun(graph, false, false);
}

std::vector<std::string> GetStringVector(std::vector<Node *> node_vector) {
  std::vector<std::string> out_vector;
  for (auto i : node_vector) {
    out_vector.push_back(i->Name());
  }
  return out_vector;
}

ir::Graph *FuseAdamWPass::FuseAdamWFun(ir::Graph *graph,
                                       bool use_master_param,
                                       bool use_skip_update) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  const std::string scope_name("fuse_adamw");
  FusePassBase::Init(scope_name, graph);

  // size_t count_adamw = 0;

  // for (auto &node : graph->Nodes()) {
  //   if (node.Name() == "adamw"){
  //     count_adamw++;
  //   }
  // }

  GraphPatternDetector gpd;
  auto *beta1_pow = gpd.mutable_pattern()
                        ->NewNode(patterns::PDNodeName(scope_name, "Beta1_Pow"))
                        ->AsInput()
                        ->assert_is_op_input("adamw", "Beta1Pow");
  patterns::AdamWAct adamw_act_pattern(gpd.mutable_pattern(), "adamw_act");

  adamw_act_pattern(beta1_pow, use_master_param, use_skip_update);

  size_t found_adamw_count = 0;
  Node *first_lr, *first_adamw_op, *skip_update, *master_prarm;
  float beta1, beta2, epsilon, coeff, lr_ratio;
  bool lazy_mode, multi_precision, use_global_beta_pow, with_decay,
      replace_adamw = true;

  std::vector<Node *> beta1_pow_vector, beta2_pow_vector, grad_vector,
      master_prarm_vector, moment1_vector, moment2_vector, param_vector,
      skip_update_vector, beta1_pow_out_vector, beta2_pow_out_vector,
      moment1_out_vector, moment2_out_vector, param_out_vector;

  std::vector<Node *> beta1_pow_w_vector, beta2_pow_w_vector, grad_w_vector,
      master_prarm_w_vector, moment1_w_vector, moment2_w_vector, param_w_vector,
      skip_update_w_vector, beta1_pow_out_w_vector, beta2_pow_out_w_vector,
      moment1_out_w_vector, moment2_out_w_vector, param_out_w_vector;

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "handle AdamWAct fuse";

    GET_IR_NODE_FROM_SUBGRAPH(adamw_op, adamw, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(beta2_pow, beta2_pow, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(grad, grad, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(learning_rate, learning_rate, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(moment1, moment1, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(moment2, moment2, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(param, param, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(beta1_pow_out, beta1_pow_out, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(beta2_pow_out, beta2_pow_out, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(moment1_out, moment1_out, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(moment2_out, moment2_out, adamw_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(param_out, param_out, adamw_act_pattern);

    if (use_master_param) {
      GET_IR_NODE_FROM_SUBGRAPH(
          master_prarm_tmp, master_prarm, adamw_act_pattern);
      master_prarm = master_prarm_tmp;
    }

    if (use_skip_update) {
      GET_IR_NODE_FROM_SUBGRAPH(
          skip_update_tmp, skip_update, adamw_act_pattern);
      skip_update = skip_update_tmp;
    }

    auto adamw_op_desc = adamw_op->Op();

    with_decay = PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("with_decay"));

    if (found_adamw_count == 0) {
      GET_IR_NODE_FROM_SUBGRAPH(adamw_op, adamw, adamw_act_pattern);
      first_lr = learning_rate;
      first_adamw_op = adamw_op;
      auto adamw_op_desc = adamw_op->Op();
      beta1 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta1"));
      beta2 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta2"));
      epsilon = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("epsilon"));
      coeff = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("coeff"));
      lazy_mode = PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("lazy_mode"));
      // min_row_size_to_use_multithread = PADDLE_GET_CONST(float,
      // adamw_op_desc->GetAttr("min_row_size_to_use_multithread"));
      multi_precision =
          PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("multi_precision"));
      use_global_beta_pow =
          PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("use_global_beta_pow"));
      lr_ratio = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("lr_ratio"));

      if (lazy_mode != false) {
        // if(lazy_mode != false || lr_ratio != 1.0) {
        replace_adamw = false;
        return;
      }
    }

    if (learning_rate->Name() != first_lr->Name()) {
      replace_adamw = false;
      return;
    }

    if (!with_decay) {
      beta1_pow_vector.push_back(subgraph.at(beta1_pow));
      beta2_pow_vector.push_back(beta2_pow);
      grad_vector.push_back(grad);
      moment1_vector.push_back(moment1);
      moment2_vector.push_back(moment2);
      param_vector.push_back(param);
      beta1_pow_out_vector.push_back(beta1_pow_out);
      beta2_pow_out_vector.push_back(beta2_pow_out);
      moment1_out_vector.push_back(moment1_out);
      moment2_out_vector.push_back(moment2_out);
      param_out_vector.push_back(param_out);

      if (use_master_param) {
        master_prarm_vector.push_back(master_prarm);
      }
      if (use_skip_update) {
        skip_update_vector.push_back(skip_update);
      }
    } else {
      beta1_pow_w_vector.push_back(subgraph.at(beta1_pow));
      beta2_pow_w_vector.push_back(beta2_pow);
      grad_w_vector.push_back(grad);
      moment1_w_vector.push_back(moment1);
      moment2_w_vector.push_back(moment2);
      param_w_vector.push_back(param);
      beta1_pow_out_w_vector.push_back(beta1_pow_out);
      beta2_pow_out_w_vector.push_back(beta2_pow_out);
      moment1_out_w_vector.push_back(moment1_out);
      moment2_out_w_vector.push_back(moment2_out);
      param_out_w_vector.push_back(param_out);

      if (use_master_param) {
        master_prarm_w_vector.push_back(master_prarm);
      }
      if (use_skip_update) {
        skip_update_w_vector.push_back(skip_update);
      }
    }

    GraphSafeRemoveNodes(g, {adamw_op});
    found_adamw_count++;
  };

  gpd(graph, handler);

  if (param_vector.size() > 0 && replace_adamw) {
    OpDesc fuse_adamw_op_desc(first_adamw_op->Op()->Block());
    fuse_adamw_op_desc.SetType("multi_tensor_adam");
    fuse_adamw_op_desc.SetInput("Params", GetStringVector(param_vector));
    fuse_adamw_op_desc.SetInput("Grads", GetStringVector(grad_vector));
    fuse_adamw_op_desc.SetInput("LearningRate", {first_lr->Name()});
    fuse_adamw_op_desc.SetInput("Moments1", GetStringVector(moment1_vector));
    fuse_adamw_op_desc.SetInput("Moments2", GetStringVector(moment2_vector));
    fuse_adamw_op_desc.SetInput("Beta1Pow", GetStringVector(beta1_pow_vector));
    fuse_adamw_op_desc.SetInput("Beta2Pow", GetStringVector(beta2_pow_vector));
    fuse_adamw_op_desc.SetOutput("ParamsOut",
                                 GetStringVector(param_out_vector));
    fuse_adamw_op_desc.SetOutput("Moments1Out",
                                 GetStringVector(moment1_out_vector));
    fuse_adamw_op_desc.SetOutput("Moments2Out",
                                 GetStringVector(moment2_out_vector));
    fuse_adamw_op_desc.SetOutput("Beta1PowOut",
                                 GetStringVector(beta1_pow_out_vector));
    fuse_adamw_op_desc.SetOutput("Beta2PowOut",
                                 GetStringVector(beta2_pow_out_vector));
    fuse_adamw_op_desc.SetAttr("beta1", beta1);
    fuse_adamw_op_desc.SetAttr("beta2", beta2);
    fuse_adamw_op_desc.SetAttr("epsilon", epsilon);
    fuse_adamw_op_desc.SetAttr("chunk_size", 32 * 2048);
    fuse_adamw_op_desc.SetAttr("weight_decay", static_cast<float>(0.0));
    fuse_adamw_op_desc.SetAttr("use_adamw", false);
    fuse_adamw_op_desc.SetAttr("multi_precision", multi_precision);
    fuse_adamw_op_desc.SetAttr("use_global_beta_pow", use_global_beta_pow);

    fuse_adamw_op_desc.SetInput("MasterParams",
                                GetStringVector(master_prarm_vector));
    fuse_adamw_op_desc.SetInput("SkipUpdate",
                                GetStringVector(skip_update_vector));

    auto fuse_adamw_node = graph->CreateOpNode(&fuse_adamw_op_desc);

    IR_NODE_LINK_TO(first_lr, fuse_adamw_node);

    for (size_t i = 0; i < param_vector.size(); i++) {
      IR_NODE_LINK_TO(beta1_pow_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(beta2_pow_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(grad_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(moment1_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(moment2_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(param_vector[i], fuse_adamw_node);
      IR_NODE_LINK_TO(fuse_adamw_node, beta1_pow_out_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_node, beta2_pow_out_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_node, moment1_out_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_node, moment2_out_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_node, param_out_vector[i]);

      if (use_master_param) {
        IR_NODE_LINK_TO(master_prarm_vector[i], fuse_adamw_node);
      }
      if (use_skip_update) {
        IR_NODE_LINK_TO(skip_update_vector[i], fuse_adamw_node);
      }
    }
  }

  if (param_w_vector.size() > 0 && replace_adamw) {
    OpDesc fuse_adamw_w_op_desc(first_adamw_op->Op()->Block());
    fuse_adamw_w_op_desc.SetType("multi_tensor_adam");
    fuse_adamw_w_op_desc.SetInput("Params", GetStringVector(param_w_vector));
    fuse_adamw_w_op_desc.SetInput("Grads", GetStringVector(grad_w_vector));
    fuse_adamw_w_op_desc.SetInput("LearningRate", {first_lr->Name()});
    fuse_adamw_w_op_desc.SetInput("Moments1",
                                  GetStringVector(moment1_w_vector));
    fuse_adamw_w_op_desc.SetInput("Moments2",
                                  GetStringVector(moment2_w_vector));
    fuse_adamw_w_op_desc.SetInput("Beta1Pow",
                                  GetStringVector(beta1_pow_w_vector));
    fuse_adamw_w_op_desc.SetInput("Beta2Pow",
                                  GetStringVector(beta2_pow_w_vector));
    fuse_adamw_w_op_desc.SetOutput("ParamsOut",
                                   GetStringVector(param_out_w_vector));
    fuse_adamw_w_op_desc.SetOutput("Moments1Out",
                                   GetStringVector(moment1_out_w_vector));
    fuse_adamw_w_op_desc.SetOutput("Moments2Out",
                                   GetStringVector(moment2_out_w_vector));
    fuse_adamw_w_op_desc.SetOutput("Beta1PowOut",
                                   GetStringVector(beta1_pow_out_w_vector));
    fuse_adamw_w_op_desc.SetOutput("Beta2PowOut",
                                   GetStringVector(beta2_pow_out_w_vector));
    fuse_adamw_w_op_desc.SetAttr("beta1", beta1);
    fuse_adamw_w_op_desc.SetAttr("beta2", beta2);
    fuse_adamw_w_op_desc.SetAttr("epsilon", epsilon);
    fuse_adamw_w_op_desc.SetAttr("chunk_size", 32 * 2048);
    fuse_adamw_w_op_desc.SetAttr("weight_decay", coeff);
    fuse_adamw_w_op_desc.SetAttr("use_adamw", true);
    fuse_adamw_w_op_desc.SetAttr("multi_precision", multi_precision);
    fuse_adamw_w_op_desc.SetAttr("use_global_beta_pow", use_global_beta_pow);

    fuse_adamw_w_op_desc.SetInput("MasterParams",
                                  GetStringVector(master_prarm_w_vector));
    fuse_adamw_w_op_desc.SetInput("SkipUpdate",
                                  GetStringVector(skip_update_w_vector));

    auto fuse_adamw_w_node = graph->CreateOpNode(&fuse_adamw_w_op_desc);

    IR_NODE_LINK_TO(first_lr, fuse_adamw_w_node);

    for (size_t i = 0; i < param_w_vector.size(); i++) {
      IR_NODE_LINK_TO(beta1_pow_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(beta2_pow_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(grad_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(moment1_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(moment2_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(param_w_vector[i], fuse_adamw_w_node);
      IR_NODE_LINK_TO(fuse_adamw_w_node, beta1_pow_out_w_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_w_node, beta2_pow_out_w_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_w_node, moment1_out_w_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_w_node, moment2_out_w_vector[i]);
      IR_NODE_LINK_TO(fuse_adamw_w_node, param_out_w_vector[i]);

      if (use_master_param) {
        IR_NODE_LINK_TO(master_prarm_w_vector[i], fuse_adamw_w_node);
      }
      if (use_skip_update) {
        IR_NODE_LINK_TO(skip_update_w_vector[i], fuse_adamw_w_node);
      }
    }
  }

  VLOG(4) << "replace adamw with fuse_adamw";

  AddStatis(found_adamw_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_adamw_op_pass, paddle::framework::ir::FuseAdamWPass);
