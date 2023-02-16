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

std::vector<std::string> GetNodeNames(const std::vector<Node *> node_vector) {
  std::vector<std::string> out_vector;
  for (auto i : node_vector) {
    out_vector.emplace_back(i->Name());
  }
  return out_vector;
}

Node *GetInputNode(const Node *op, const std::string name) {
  Node *out = nullptr;

  for (auto &node : op->inputs) {
    if (node->Name() == op->Op()->Input(name)[0]) {
      out = node;
      break;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      out, platform::errors::InvalidArgument("Input's name cannot be found."));

  return out;
}

Node *GetOutputNode(const Node *op, const std::string name) {
  Node *out = nullptr;

  for (auto &node : op->outputs) {
    if (node->Name() == op->Op()->Output(name)[0]) {
      out = node;
      break;
    }
  }

  PADDLE_ENFORCE_NOT_NULL(
      out, platform::errors::InvalidArgument("Output's name cannot be found."));

  return out;
}

void SaveInOutNodes(
    std::vector<std::vector<std::vector<Node *>>> *inout_vectors,
    const std::vector<std::string> inputs_name,
    const std::vector<std::string> outputs_name,
    const Node *op,
    size_t condition) {
  size_t i = 0;

  for (auto &name : inputs_name) {
    (*inout_vectors)[condition][i].emplace_back(GetInputNode(op, name));
    i++;
  }
  for (auto &name : outputs_name) {
    (*inout_vectors)[condition][i].emplace_back(GetOutputNode(op, name));
    i++;
  }
  if (condition & 1) {
    (*inout_vectors)[condition][i++].emplace_back(
        GetInputNode(op, "MasterParam"));
    (*inout_vectors)[condition][i].emplace_back(
        GetOutputNode(op, "MasterParamOut"));
  }
}

void InsertOp(Node *first_lr,
              Node *first_skip_update,
              const std::vector<std::vector<std::vector<Node *>>> inout_vectors,
              const std::vector<std::string> inputs_name,
              const std::vector<std::string> outputs_name,
              paddle::framework::BlockDesc *block,
              float beta1,
              float beta2,
              float epsilon,
              float coeff,
              int op_role,
              bool use_global_beta_pow,
              bool replace_adamw,
              bool use_skip_update,
              ir::Graph *graph) {
  for (size_t condition = 0; condition < 4; condition++) {
    float weight_decay = static_cast<float>(0.0);
    bool use_adamw = false;

    if (condition & 2) {
      weight_decay = coeff;
      use_adamw = true;
    }
    if (inout_vectors[condition][0].size() > 0 && replace_adamw) {
      OpDesc fuse_adamw_op_desc(block);
      fuse_adamw_op_desc.SetType("multi_tensor_adam");

      size_t i = 0;

      for (auto &name : inputs_name) {
        fuse_adamw_op_desc.SetInput(name,
                                    GetNodeNames(inout_vectors[condition][i]));
        i++;
      }

      fuse_adamw_op_desc.SetInput("LearningRate", {first_lr->Name()});
      if (use_skip_update) {
        fuse_adamw_op_desc.SetInput("SkipUpdate", {first_skip_update->Name()});
      } else {
        fuse_adamw_op_desc.SetInput("SkipUpdate", {});
      }

      for (auto &name : outputs_name) {
        fuse_adamw_op_desc.SetOutput(name,
                                     GetNodeNames(inout_vectors[condition][i]));
        i++;
      }

      if (condition & 1) {
        fuse_adamw_op_desc.SetInput(
            "MasterParams", GetNodeNames(inout_vectors[condition][i++]));
        fuse_adamw_op_desc.SetOutput("MasterParamsOut",
                                     GetNodeNames(inout_vectors[condition][i]));
      } else {
        fuse_adamw_op_desc.SetInput("MasterParams", {});
      }

      fuse_adamw_op_desc.SetAttr("beta1", beta1);
      fuse_adamw_op_desc.SetAttr("beta2", beta2);
      fuse_adamw_op_desc.SetAttr("op_role", op_role);
      fuse_adamw_op_desc.SetAttr("epsilon", epsilon);
      fuse_adamw_op_desc.SetAttr("chunk_size", 16 * 2048);
      fuse_adamw_op_desc.SetAttr("weight_decay", weight_decay);
      fuse_adamw_op_desc.SetAttr("use_adamw", use_adamw);
      fuse_adamw_op_desc.SetAttr("multi_precision",
                                 condition & 1 ? true : false);
      fuse_adamw_op_desc.SetAttr("use_global_beta_pow", use_global_beta_pow);

      auto fuse_adamw_node = graph->CreateOpNode(&fuse_adamw_op_desc);

      IR_NODE_LINK_TO(first_lr, fuse_adamw_node);
      if (use_skip_update) {
        IR_NODE_LINK_TO(first_skip_update, fuse_adamw_node);
      }

      for (size_t k = 0; k < inout_vectors[condition][0].size(); k++) {
        size_t j = 0;

        for (; j < inputs_name.size(); j++) {
          IR_NODE_LINK_TO(inout_vectors[condition][j][k], fuse_adamw_node);
        }
        for (; j < inputs_name.size() + outputs_name.size(); j++) {
          IR_NODE_LINK_TO(fuse_adamw_node, inout_vectors[condition][j][k]);
        }
        if (condition & 1) {
          IR_NODE_LINK_TO(inout_vectors[condition][j][k], fuse_adamw_node);
          j++;
          IR_NODE_LINK_TO(fuse_adamw_node, inout_vectors[condition][j][k]);
        }
      }
    }
  }
}

void FuseAdamWPass::ApplyImpl(ir::Graph *graph) const {
  graph = FuseAdamWFun(graph);
}

ir::Graph *FuseAdamWPass::FuseAdamWFun(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));

  VLOG(4) << "handle fuse AdadW";

  const std::string scope_name("fuse_adamw");
  FusePassBase::Init(scope_name, graph);

  size_t found_adamw_count = 0;

  Node *first_lr = nullptr, *first_skip_update = nullptr;
  int op_role = 0;
  float beta1 = 0.9, beta2 = 0.999, epsilon = 1.0e-8, first_coeff = 0.0,
        coeff = 0.0, lr_ratio = 1.0;
  bool lazy_mode = false, multi_precision = false, use_global_beta_pow = false,
       with_decay = false, replace_adamw = true, use_skip_update = false;
  int64_t min_row_size_to_use_multithread = 1000;
  paddle::framework::BlockDesc *block = nullptr;
  size_t condition = 0;

  const std::vector<std::string> inputs_name = {
      "Param", "Grad", "Moment1", "Moment2", "Beta1Pow", "Beta2Pow"};

  const std::vector<std::string> outputs_name = {
      "ParamOut", "Moment1Out", "Moment2Out", "Beta1PowOut", "Beta2PowOut"};

  const std::vector<std::string> replace_inputs_name = {
      "Params", "Grads", "Moments1", "Moments2", "Beta1Pows", "Beta2Pows"};

  const std::vector<std::string> repalce_outputs_name = {"ParamsOut",
                                                         "Moments1Out",
                                                         "Moments2Out",
                                                         "Beta1PowsOut",
                                                         "Beta2PowsOut"};

  std::vector<std::vector<std::vector<Node *>>> inout_vectors(
      4, std::vector<std::vector<Node *>>(13));

  std::unordered_set<const Node *> adamw_op_del_set;

  for (auto &node : graph->Nodes()) {
    if (node->Name() == "adamw") {
      const Node *adamw_op = node;
      Node *skip_update = nullptr;
      Node *learning_rate = GetInputNode(adamw_op, "LearningRate");
      auto adamw_op_desc = adamw_op->Op();

      // Get skip_update and coeff, these wiil be used to cheak whether we can
      // use fuse_adamw.
      for (auto &node : adamw_op->inputs) {
        auto in_name = adamw_op_desc->Input("SkipUpdate");
        if (!in_name.empty()) {
          if (node->Name() == in_name[0]) {
            use_skip_update = true;
            skip_update = node;
            break;
          }
        }
      }
      coeff = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("coeff"));

      // Get with_decay and multi_precision, there are 4 conditions: whether to
      // use with_decay and whether to use multi_precision.
      with_decay = PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("with_decay"));
      multi_precision =
          PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("multi_precision"));
      condition = with_decay << 1;
      condition += multi_precision;

      // Get attrs and block
      if (found_adamw_count == 0) {
        // Get blokc
        block = adamw_op_desc->Block();
        // Get attrs
        first_lr = learning_rate;
        beta1 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta1"));
        beta2 = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("beta2"));
        op_role = PADDLE_GET_CONST(int, adamw_op_desc->GetAttr("op_role"));
        epsilon = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("epsilon"));
        first_coeff = coeff;
        lazy_mode = PADDLE_GET_CONST(bool, adamw_op_desc->GetAttr("lazy_mode"));
        min_row_size_to_use_multithread = PADDLE_GET_CONST(
            int64_t, adamw_op_desc->GetAttr("min_row_size_to_use_multithread"));
        use_global_beta_pow = PADDLE_GET_CONST(
            bool, adamw_op_desc->GetAttr("use_global_beta_pow"));
        lr_ratio = PADDLE_GET_CONST(float, adamw_op_desc->GetAttr("lr_ratio"));
        if (use_skip_update) {
          first_skip_update = skip_update;
        }
        // We do not support these conditions
        if (lazy_mode != false || lr_ratio != 1.0 ||
            min_row_size_to_use_multithread != 1000) {
          replace_adamw = false;
          return graph;
        }
      }

      // We do not support these conditions
      if ((learning_rate->Name() != first_lr->Name()) ||
          (coeff != first_coeff) ||
          (use_skip_update &&
           skip_update->Name() != first_skip_update->Name())) {
        replace_adamw = false;
        return graph;
      }

      adamw_op_del_set.insert(adamw_op);

      // Save input and output Nodes
      SaveInOutNodes(
          &inout_vectors, inputs_name, outputs_name, adamw_op, condition);

      found_adamw_count++;
    }
  }

  // Remove old op
  if (replace_adamw &&
      (inout_vectors[0][0].size() > 0 || inout_vectors[1][0].size() > 0 ||
       inout_vectors[2][0].size() > 0 || inout_vectors[3][0].size() > 0)) {
    // for (auto adamw_op : adamw_op_del_set) {
    //   GraphSafeRemoveNodes(graph, {adamw_op});
    // }
    GraphSafeRemoveNodes(graph, adamw_op_del_set);
  }

  // Insert new op
  InsertOp(first_lr,
           first_skip_update,
           inout_vectors,
           replace_inputs_name,
           repalce_outputs_name,
           block,
           beta1,
           beta2,
           epsilon,
           coeff,
           op_role,
           use_global_beta_pow,
           replace_adamw,
           use_skip_update,
           graph);

  VLOG(4) << "replace adamw with fuse_adamw";

  AddStatis(found_adamw_count);
  return graph;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_adamw_op_pass, paddle::framework::ir::FuseAdamWPass);
