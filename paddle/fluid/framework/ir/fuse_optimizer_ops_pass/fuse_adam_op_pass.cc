//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <sys/types.h>

#include <string>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/fuse_optimizer_ops_pass/fuse_optimizer_op_pass.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework::ir {

class Node;

class FuseAdamOpPass : public FuseOptimizerOpPass {
 private:
  const std::string GetOpType() const override { return "adam"; }

  const std::vector<std::string> GetAuxiliaryVarNames() const override {
    return {"Moment1", "Moment2", "Beta1Pow", "Beta2Pow"};
  }

  ir::Node *FuseOptimizerOps(
      const std::unordered_map<std::string, std::vector<std::string>>
          &aux_var_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &adam_ops,
      ir::Graph *graph) const override {
    auto fused_adam_node =
        FuseAdamOps(aux_var_set, fused_vars_name, adam_ops, graph);
    return fused_adam_node;
  }

  void RemoveCycleDepsBetweenOpNodes(Graph *graph,
                                     const Node *fused_scale1,
                                     const Node *fused_scale2) const {
    std::unordered_set<Node *> not_need_ctrl_var_nodes;
    std::unordered_set<Node *> fused_scale2_in_nodes;
    fused_scale2_in_nodes.insert(fused_scale2->inputs.begin(),
                                 fused_scale2->inputs.end());
    for (auto &out_node : fused_scale1->outputs) {
      if (fused_scale2_in_nodes.count(out_node)) {
        PADDLE_ENFORCE_EQ(out_node->IsCtrlVar(),
                          true,
                          common::errors::PreconditionNotMet(
                              "In adam op pass, the dependency var(%s) only "
                              "should be ctrl var.",
                              out_node->Name()));
        not_need_ctrl_var_nodes.insert(out_node);
      }
    }

    for (auto &node : not_need_ctrl_var_nodes) {
      // remove this node from the input op node.
      PADDLE_ENFORCE_EQ(
          node->inputs.empty(),
          false,
          common::errors::PreconditionNotMet(
              "Node(%s)'s input should not be empty here.", node->Name()));
      auto op_node = node->inputs.front();
      PADDLE_ENFORCE_EQ(op_node->IsOp(),
                        true,
                        common::errors::PreconditionNotMet(
                            "Node(%s) should be an OP node.", op_node->Name()));
      op_node->outputs.erase(remove_if(op_node->outputs.begin(),
                                       op_node->outputs.end(),
                                       [&node](const Node *op_out_node) {
                                         return op_out_node == node;
                                       }),
                             op_node->outputs.end());

      // remove this node from the output op nodes.
      for (auto &out_op_node : node->outputs) {
        out_op_node->inputs.erase(remove_if(out_op_node->inputs.begin(),
                                            out_op_node->inputs.end(),
                                            [&node](const Node *op_in_node) {
                                              return op_in_node == node;
                                            }),
                                  out_op_node->inputs.end());
      }

      graph->RemoveNode(node);
    }
  }

  ir::Node *FuseAdamOps(
      const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
      const std::unordered_map<std::string, std::string> &fused_vars_name,
      const std::vector<ir::Node *> &adam_ops,
      ir::Graph *graph) const {
    PADDLE_ENFORCE_GT(
        adam_ops.size(),
        static_cast<size_t>(0),
        common::errors::InvalidArgument("No adam op in the graph."));

    // Check attributions
    // NOTE: If new attribution is added, the following code maybe need change.
    int op_role = PADDLE_GET_CONST(
        int,
        adam_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    float beta1 = PADDLE_GET_CONST(float, adam_ops[0]->Op()->GetAttr("beta1"));
    float beta2 = PADDLE_GET_CONST(float, adam_ops[0]->Op()->GetAttr("beta2"));
    float epsilon =
        PADDLE_GET_CONST(float, adam_ops[0]->Op()->GetAttr("epsilon"));
    bool lazy_mode =
        PADDLE_GET_CONST(bool, adam_ops[0]->Op()->GetAttr("lazy_mode"));
    int64_t min_row_size_to_use_multithread = PADDLE_GET_CONST(
        int64_t, adam_ops[0]->Op()->GetAttr("min_row_size_to_use_multithread"));
    for (auto &adam_op : adam_ops) {
      PADDLE_ENFORCE_EQ(
          beta1,
          PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("beta1")),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(beta1) must be same, but there are two "
              "different "
              "value: %f, %f.",
              beta1,
              PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("beta1"))));
      PADDLE_ENFORCE_EQ(
          beta2,
          PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("beta2")),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(beta2) must be same, but there are two "
              "different "
              "value: %f, %f.",
              beta2,
              PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("beta2"))));
      PADDLE_ENFORCE_EQ(
          epsilon,
          PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("epsilon")),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(epsilon) must be same, but there are two "
              "different "
              "value: %f, %f.",
              epsilon,
              PADDLE_GET_CONST(float, adam_op->Op()->GetAttr("epsilon"))));
      PADDLE_ENFORCE_EQ(
          lazy_mode,
          PADDLE_GET_CONST(bool, adam_op->Op()->GetAttr("lazy_mode")),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(lazy_mode) must be same, but there are two "
              "different "
              "value: %d, %d.",
              lazy_mode,
              PADDLE_GET_CONST(bool, adam_op->Op()->GetAttr("lazy_mode"))));
      PADDLE_ENFORCE_EQ(
          min_row_size_to_use_multithread,
          PADDLE_GET_CONST(
              int64_t,
              adam_op->Op()->GetAttr("min_row_size_to_use_multithread")),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(min_row_size_to_use_multithread) must be "
              "same, but there are two different value: %I64, %I64.",
              min_row_size_to_use_multithread,
              PADDLE_GET_CONST(
                  int64_t,
                  adam_op->Op()->GetAttr("min_row_size_to_use_multithread"))));
      PADDLE_ENFORCE_EQ(
          op_role,
          PADDLE_GET_CONST(
              int,
              adam_op->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName())),
          common::errors::PreconditionNotMet(
              "All adam Op's attr(op_role) must be same, but there are two "
              "different "
              "value: %d, %d.",
              op_role,
              PADDLE_GET_CONST(int,
                               adam_op->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName()))));
    }

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var node.

    VLOG(6) << "Insert adam to graph ";
    OpDesc adam_desc(adam_ops[0]->Op()->Block());
    adam_desc.SetType("adam");
    adam_desc.SetInput(kParam, {fused_vars_name.at(kParam)});
    adam_desc.SetInput(kGrad, {fused_vars_name.at(kGrad)});
    adam_desc.SetInput("Moment1", {fused_vars_name.at("Moment1")});
    adam_desc.SetInput("Moment2", {fused_vars_name.at("Moment2")});
    // TODO(zcd): The LearningRate, Beta1Pow, Beta2Pow should be equal.
    adam_desc.SetInput(kLearningRate, adam_ops[0]->Op()->Input(kLearningRate));
    adam_desc.SetInput("Beta1Pow", adam_ops[0]->Op()->Input("Beta1Pow"));
    adam_desc.SetInput("Beta2Pow", adam_ops[0]->Op()->Input("Beta2Pow"));

    adam_desc.SetOutput("ParamOut", {fused_vars_name.at(kParam)});
    adam_desc.SetOutput("Moment1Out", {fused_vars_name.at("Moment1")});
    adam_desc.SetOutput("Moment2Out", {fused_vars_name.at("Moment2")});
    adam_desc.SetOutput("Beta1PowOut", {fused_vars_name.at("Beta1Pow")});
    adam_desc.SetOutput("Beta2PowOut", {fused_vars_name.at("Beta2Pow")});
    adam_desc.SetAttr("beta1", beta1);
    adam_desc.SetAttr("beta2", beta2);
    adam_desc.SetAttr("epsilon", epsilon);
    adam_desc.SetAttr("lazy_mode", lazy_mode);
    adam_desc.SetAttr("min_row_size_to_use_multithread",
                      min_row_size_to_use_multithread);
    adam_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
    return graph->CreateOpNode(&adam_desc);
  }

  ir::Node *FuseScaleOps(const std::vector<std::string> &beta_name,
                         const std::string &fused_var_name,
                         const std::vector<ir::Node *> &adam_ops,
                         ir::Graph *graph) const {
    PADDLE_ENFORCE_EQ(beta_name.size(),
                      adam_ops.size(),
                      common::errors::InvalidArgument(
                          "Beta name size(%d) must equal to adam op size(%d).",
                          beta_name.size(),
                          adam_ops.size()));
    const std::string scale_op_name = "scale";

    // Get the scale_ops of dealing the adam's beta var.
    std::vector<ir::Node *> scale_ops;
    scale_ops.reserve(beta_name.size());
    for (size_t i = 0; i < adam_ops.size(); ++i) {
      auto &beta_1_pow_name = beta_name[i];
      auto beta_pow_iter =
          std::find_if(adam_ops[i]->inputs.begin(),
                       adam_ops[i]->inputs.end(),
                       [&beta_1_pow_name](ir::Node *var_node) -> bool {
                         return var_node->Var() &&
                                var_node->Var()->Name() == beta_1_pow_name;
                       });
      PADDLE_ENFORCE_NE(beta_pow_iter,
                        adam_ops[i]->inputs.end(),
                        common::errors::NotFound("Can not find %s in adam ops.",
                                                 beta_1_pow_name));

      auto beta_pow_node = *beta_pow_iter;
      auto scale_op_iter = std::find_if(
          beta_pow_node->outputs.begin(),
          beta_pow_node->outputs.end(),
          [&scale_op_name](ir::Node *op_node) -> bool {
            return op_node->Op() && op_node->Op()->Type() == scale_op_name;
          });
      PADDLE_ENFORCE_NE(
          scale_op_iter,
          beta_pow_node->outputs.end(),
          common::errors::NotFound("Can not find %s in beta pow node.",
                                   scale_op_name));

      scale_ops.emplace_back(*scale_op_iter);
    }
    PADDLE_ENFORCE_EQ(
        scale_ops.size(),
        beta_name.size(),
        common::errors::PreconditionNotMet(
            "Beta name size(%d) must equal to scale ops size(%d).",
            beta_name.size(),
            scale_ops.size()));
    VLOG(6) << "The number of scale op is " << scale_ops.size() << ".";
    // Check attributions
    // NOTE: If new attribution is added, the following code maybe need change.
    int op_role = PADDLE_GET_CONST(
        int,
        scale_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
    float scale = PADDLE_GET_CONST(float, scale_ops[0]->Op()->GetAttr("scale"));
    float bias = PADDLE_GET_CONST(float, scale_ops[0]->Op()->GetAttr("bias"));
    bool bias_after_scale =
        PADDLE_GET_CONST(bool, scale_ops[0]->Op()->GetAttr("bias_after_scale"));
    for (auto &scale_op : scale_ops) {
      PADDLE_ENFORCE_EQ(
          scale,
          PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale")),
          common::errors::PreconditionNotMet(
              "All scale Op's attr(scale) must be same, but there are two "
              "different "
              "value: %f, %f.",
              scale,
              PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("scale"))));
      PADDLE_ENFORCE_EQ(
          bias,
          PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("bias")),
          common::errors::PreconditionNotMet(
              "All scale Op's attr(bias) must be same, but there are two "
              "different "
              "value: %f, %f.",
              bias,
              PADDLE_GET_CONST(float, scale_op->Op()->GetAttr("bias"))));
      PADDLE_ENFORCE_EQ(
          bias_after_scale,
          PADDLE_GET_CONST(bool, scale_op->Op()->GetAttr("bias_after_scale")),
          common::errors::PreconditionNotMet(
              "All scale Op's attr(bias_after_scale) must be same, but there "
              "are two different value: %d, %d.",
              bias_after_scale,
              PADDLE_GET_CONST(bool,
                               scale_op->Op()->GetAttr("bias_after_scale"))));
      PADDLE_ENFORCE_EQ(
          op_role,
          PADDLE_GET_CONST(int,
                           scale_op->Op()->GetAttr(
                               OpProtoAndCheckerMaker::OpRoleAttrName())),
          common::errors::PreconditionNotMet(
              "All scale Op's attr(op_role) must be same, but there are two "
              "different "
              "value: %d, %d.",
              op_role,
              PADDLE_GET_CONST(int,
                               scale_op->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName()))));
    }

    // NOTE: fused_var is only exist in scope, so the graph doesn't have
    // fused_var node.

    VLOG(6) << "Insert fused scale to graph.";
    OpDesc scale_desc(scale_ops[0]->Op()->Block());
    scale_desc.SetType("scale");
    scale_desc.SetInput("X", {fused_var_name});
    scale_desc.SetOutput("Out", {fused_var_name});
    scale_desc.SetAttr("scale", scale);
    scale_desc.SetAttr("bias", bias);
    scale_desc.SetAttr("bias_after_scale", bias_after_scale);
    scale_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
    auto scale_node = graph->CreateOpNode(&scale_desc);

    InsertInputAndOutputForFusedOpNode(scale_ops, graph, scale_node);
    // Delete scale_ops
    for (auto &scale_op : scale_ops) {
      graph->RemoveNode(scale_op);
    }
    return scale_node;
  }
};
}  // namespace paddle::framework::ir

REGISTER_PASS(fuse_adam_op_pass, paddle::framework::ir::FuseAdamOpPass);
