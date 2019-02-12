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

#include "paddle/fluid/framework/details/fuse_adam_op_pass.h"
#include <algorithm>
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> FuseAdamOpPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  ir::Graph &result = *graph;

  const std::string fuse_op_type = "adam";
  std::vector<std::string> aux_var_names = {"Param", "Moment1", "Moment2",
                                            "Beta1Pow", "Beta2Pow"};

  // Step 1: Get the specified op and auxiliary variables.
  std::unordered_map<std::string, std::vector<std::string>> aux_var_set;
  std::vector<ir::Node *> adam_ops;
  for (ir::Node *node : result.Nodes()) {
    if (node->IsOp()) {
      GetSpecifiedOpsAndVars(fuse_op_type, aux_var_names, node, &adam_ops,
                             &aux_var_set);
    }
  }

  PADDLE_ENFORCE_NE(adam_ops.size(), 0, "Not found %s.", fuse_op_type);
  VLOG(10) << "Find " << fuse_op_type << " operators: " << adam_ops.size();

  // Step 2: Insert fused_var_name to FusedVars, and the FusedVars need be
  // initialized in scopes before execution.
  if (!result.Has(kFusedVars)) {
    result.Set(kFusedVars, new FusedVars);
  }

  std::unordered_map<std::string, std::string> fused_vars_name;
  fused_vars_name.reserve(aux_var_names.size() + 1);
  const std::string prefix(kGradVarSuffix);
  for (auto &var_name : aux_var_names) {
    auto fused_var_name = prefix + "_" + fuse_op_type + "_" + var_name;
    fused_vars_name.emplace(var_name, fused_var_name);
    result.Get<FusedVars>(kFusedVars).emplace_back(fused_var_name);
  }

  // Step 3: Sort the parameters and auxiliary variables according
  // to parameters' name to make variables' name correspond correctly.
  SortVarsName(aux_var_names[0], &aux_var_set, &adam_ops);

  // Step 4: Alloc continuous space for Moment1, Moment2, Beta1Pow, Beta2Pow
  // of all the adam ops separately.
  // And alloc_continuous_space ops are placed in RunOnlyOnceProgram,
  // which is executed before running the model with ParallelExecutor.
  if (!result.Has(kRunOnlyOnceProgram)) {
    result.Set(kRunOnlyOnceProgram, new RunOnlyOnceProgram);
  }
  result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).emplace_back();
  auto &program_desc =
      result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).back();
  auto *global_block = program_desc.MutableBlock(0);
  for (auto &var_name : aux_var_names) {
    AppendAllocContinuousSpace(aux_var_set.at(var_name),
                               fused_vars_name.at(var_name), true,
                               global_block);
  }

  // Step 5: Get the fused Gradient's name
  auto fused_grad = prefix + "_GRAD";
  auto &fused_vars = result.Get<FusedVars>(kFusedVars);
  auto iter = std::find(fused_vars.begin(), fused_vars.end(), fused_grad);
  PADDLE_ENFORCE(iter != fused_vars.end(), "Not find the fused_grad.");
  fused_vars_name.emplace("Grad", fused_grad);

  // Step 6: Fuse Adam Ops and Scale Ops
  FuseAdamOps(aux_var_set, fused_vars_name, adam_ops, &result);
  FuseScaleOps(aux_var_set.at("Beta1Pow"), fused_vars_name.at("Beta1Pow"),
               adam_ops, &result);
  FuseScaleOps(aux_var_set.at("Beta2Pow"), fused_vars_name.at("Beta2Pow"),
               adam_ops, &result);

  // Step 7: Remove Adam Ops
  for (auto &adam_op : adam_ops) {
    graph->RemoveNode(adam_op);
  }

  return std::move(graph);
}

void FuseAdamOpPass::SortVarsName(
    const std::string &str,
    std::unordered_map<std::string, std::vector<std::string>> *aux_vars_set,
    std::vector<ir::Node *> *ops) const {
  auto &param_vec = aux_vars_set->at(str);
  std::vector<size_t> param_sort_idx;
  param_sort_idx.reserve(param_vec.size());
  for (size_t i = 0; i < param_vec.size(); ++i) {
    param_sort_idx.emplace_back(i);
  }

  std::sort(param_sort_idx.begin(), param_sort_idx.end(),
            [&param_vec](size_t a, size_t b) -> bool {
              return param_vec[a] < param_vec[b];
            });

  for (auto &aux_vars : *aux_vars_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i = 0; i < aux_vars.second.size(); ++i) {
      sorted_vars.emplace_back(aux_vars.second.at(param_sort_idx[i]));
    }
    std::swap(aux_vars.second, sorted_vars);
  }

  std::vector<ir::Node *> sorted_ops;
  sorted_ops.reserve(ops->size());
  for (size_t i = 0; i < ops->size(); ++i) {
    sorted_ops.emplace_back(ops->at(param_sort_idx[i]));
  }
  std::swap(*ops, sorted_ops);
}

void FuseAdamOpPass::GetSpecifiedOpsAndVars(
    const std::string &op_type, const std::vector<std::string> &aux_vars_name,
    ir::Node *node, std::vector<ir::Node *> *ops,
    std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
    const {
  if (node->Op()->Type() != op_type) return;

  for (auto &var_n : aux_vars_name) {
    auto arg_names = node->Op()->Input(var_n);
    PADDLE_ENFORCE_EQ(arg_names.size(), 1);
    (*aux_args_name)[var_n].emplace_back(arg_names[0]);
  }
  ops->emplace_back(node);
}

void FuseAdamOpPass::FuseAdamOps(
    const std::unordered_map<std::string, std::vector<std::string>> &vars_set,
    const std::unordered_map<std::string, std::string> &fused_vars_name,
    const std::vector<ir::Node *> &adam_ops, ir::Graph *graph) const {
  PADDLE_ENFORCE_GT(adam_ops.size(), 0);

  // Check attributions
  // NOTE: If new attribution is added, the following code maybe need change.
  int op_role = boost::get<int>(
      adam_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
  float beta1 = boost::get<float>(adam_ops[0]->Op()->GetAttr("beta1"));
  float beta2 = boost::get<float>(adam_ops[0]->Op()->GetAttr("beta2"));
  float epsilon = boost::get<float>(adam_ops[0]->Op()->GetAttr("epsilon"));
  bool lazy_mode = boost::get<bool>(adam_ops[0]->Op()->GetAttr("lazy_mode"));
  int64_t min_row_size_to_use_multithread = boost::get<int64_t>(
      adam_ops[0]->Op()->GetAttr("min_row_size_to_use_multithread"));
  for (auto &adam_op : adam_ops) {
    PADDLE_ENFORCE_EQ(beta1,
                      boost::get<float>(adam_op->Op()->GetAttr("beta1")));
    PADDLE_ENFORCE_EQ(beta2,
                      boost::get<float>(adam_op->Op()->GetAttr("beta2")));
    PADDLE_ENFORCE_EQ(epsilon,
                      boost::get<float>(adam_op->Op()->GetAttr("epsilon")));
    PADDLE_ENFORCE_EQ(lazy_mode,
                      boost::get<bool>(adam_op->Op()->GetAttr("lazy_mode")));
    PADDLE_ENFORCE_EQ(min_row_size_to_use_multithread,
                      boost::get<int64_t>(adam_op->Op()->GetAttr(
                          "min_row_size_to_use_multithread")));
    PADDLE_ENFORCE_EQ(op_role, boost::get<int>(adam_op->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())));
  }

  // NOTE: fused_var is only exist in scope, so the graph doesn't have fused_var
  // node.

  // Add reset_dim, only fuse the scale ops
  VLOG(10) << "Insert adam to graph ";
  //  int num_scale_op = static_cast<int>(beta_name.size());
  // Add fused scale
  OpDesc adam_desc;
  adam_desc.SetType("adam");
  adam_desc.SetInput("Param", {fused_vars_name.at("Param")});
  adam_desc.SetInput("Grad", {fused_vars_name.at("Grad")});
  adam_desc.SetInput("Moment1", {fused_vars_name.at("Moment1")});
  adam_desc.SetInput("Moment2", {fused_vars_name.at("Moment2")});
  // TODO(zcd): The LearningRate, Beta1Pow, Beta2Pow should be equal.
  adam_desc.SetInput("LearningRate", adam_ops[0]->Op()->Input("LearningRate"));
  adam_desc.SetInput("Beta1Pow", adam_ops[0]->Op()->Input("Beta1Pow"));
  adam_desc.SetInput("Beta2Pow", adam_ops[0]->Op()->Input("Beta2Pow"));

  adam_desc.SetOutput("ParamOut", {fused_vars_name.at("Param")});
  adam_desc.SetOutput("Moment1Out", {fused_vars_name.at("Moment1")});
  adam_desc.SetOutput("Moment2Out", {fused_vars_name.at("Moment2")});
  adam_desc.SetAttr("beta1", beta1);
  adam_desc.SetAttr("beta2", beta2);
  adam_desc.SetAttr("epsilon", epsilon);
  adam_desc.SetAttr("lazy_mode", lazy_mode);
  adam_desc.SetAttr("min_row_size_to_use_multithread",
                    min_row_size_to_use_multithread);
  // NOTE: multi_devices_pass requires that every op should have a role.
  adam_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);

  auto adam_node = graph->CreateOpNode(&adam_desc);

  for (auto adam_op : adam_ops) {
    // set inputs
    adam_node->inputs.insert(adam_node->inputs.begin(), adam_op->inputs.begin(),
                             adam_op->inputs.end());
    for (auto &input : adam_op->inputs) {
      std::replace(input->outputs.begin(), input->outputs.end(), adam_op,
                   adam_node);
    }
    // set outputs
    adam_node->outputs.insert(adam_node->outputs.begin(),
                              adam_op->outputs.begin(), adam_op->outputs.end());
    for (auto &output : adam_op->outputs) {
      std::replace(output->inputs.begin(), output->inputs.end(), adam_op,
                   adam_node);
    }
  }
}

void FuseAdamOpPass::FuseScaleOps(const std::vector<std::string> &beta_name,
                                  const std::string &fused_var_name,
                                  const std::vector<ir::Node *> &adam_ops,
                                  ir::Graph *graph) const {
  PADDLE_ENFORCE_EQ(beta_name.size(), adam_ops.size());
  const std::string scale_op_name = "scale";

  // Get the scale_ops of dealing the adam's beta var.
  std::vector<ir::Node *> scale_ops;
  scale_ops.reserve(beta_name.size());
  for (size_t i = 0; i < adam_ops.size(); ++i) {
    auto &beta_1_pow_name = beta_name[i];
    auto beta_pow_iter = std::find_if(
        adam_ops[i]->inputs.begin(), adam_ops[i]->inputs.end(),
        [&beta_name, &beta_1_pow_name](ir::Node *var_node) -> bool {
          return var_node->Var() && var_node->Var()->Name() == beta_1_pow_name;
        });
    PADDLE_ENFORCE(beta_pow_iter != adam_ops[i]->inputs.end());

    auto beta_pow_node = *beta_pow_iter;
    auto scale_op_iter = std::find_if(
        beta_pow_node->outputs.begin(), beta_pow_node->outputs.end(),
        [&scale_op_name](ir::Node *op_node) -> bool {
          return op_node->Op() && op_node->Op()->Type() == scale_op_name;
        });
    PADDLE_ENFORCE(scale_op_iter != beta_pow_node->outputs.end());

    scale_ops.emplace_back(*scale_op_iter);
  }
  PADDLE_ENFORCE_EQ(scale_ops.size(), beta_name.size());

  // Check attributions
  // NOTE: If new attribution is added, the following code maybe need change.
  int op_role = boost::get<int>(
      scale_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));
  float scale = boost::get<float>(scale_ops[0]->Op()->GetAttr("scale"));
  float bias = boost::get<float>(scale_ops[0]->Op()->GetAttr("bias"));
  bool bias_after_scale =
      boost::get<bool>(scale_ops[0]->Op()->GetAttr("bias_after_scale"));
  for (auto &scale_op : scale_ops) {
    PADDLE_ENFORCE_EQ(scale,
                      boost::get<float>(scale_op->Op()->GetAttr("scale")));
    PADDLE_ENFORCE_EQ(bias, boost::get<float>(scale_op->Op()->GetAttr("bias")));
    PADDLE_ENFORCE_EQ(
        bias_after_scale,
        boost::get<bool>(scale_op->Op()->GetAttr("bias_after_scale")));
    PADDLE_ENFORCE_EQ(op_role, boost::get<int>(scale_op->Op()->GetAttr(
                                   OpProtoAndCheckerMaker::OpRoleAttrName())));
  }

  // NOTE: fused_var is only exist in scope, so the graph doesn't have fused_var
  // node.

  VLOG(10) << "Insert fused scale to graph.";
  OpDesc scale_desc;
  scale_desc.SetType("scale");
  scale_desc.SetInput("X", {fused_var_name});
  scale_desc.SetOutput("Out", {fused_var_name});
  scale_desc.SetAttr("scale", scale);
  scale_desc.SetAttr("bias", bias);
  scale_desc.SetAttr("bias_after_scale", bias_after_scale);
  // NOTE: multi_devices_pass requires that every op should have a role.
  scale_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
  auto scale_node = graph->CreateOpNode(&scale_desc);

  for (auto scale_op : scale_ops) {
    // set inputs
    scale_node->inputs.insert(scale_node->inputs.begin(),
                              scale_op->inputs.begin(), scale_op->inputs.end());
    for (auto &input : scale_op->inputs) {
      std::replace(input->outputs.begin(), input->outputs.end(), scale_op,
                   scale_node);
    }
    // set outputs
    scale_node->outputs.insert(scale_node->outputs.begin(),
                               scale_op->outputs.begin(),
                               scale_op->outputs.end());
    for (auto &output : scale_op->outputs) {
      std::replace(output->inputs.begin(), output->inputs.end(), scale_op,
                   scale_node);
    }
  }

  // Delete scale_ops
  for (auto &scale_op : scale_ops) {
    graph->RemoveNode(scale_op);
  }
}

void FuseAdamOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &args, const std::string &out_arg,
    bool copy_data, BlockDesc *global_block) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_continuous_space");
  op_desc->SetInput("Input", args);
  op_desc->SetOutput("Output", args);
  op_desc->SetOutput("FusedOutput", {out_arg});
  op_desc->SetAttr("copy_data", copy_data);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_adam_op_pass, paddle::framework::details::FuseAdamOpPass);
