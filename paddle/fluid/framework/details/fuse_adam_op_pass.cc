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

  const std::string fuse_op = "adam";
  std::vector<std::string> aux_var_names = {"Param", "Moment1", "Moment2",
                                            "Beta1Pow", "Beta2Pow"};

  std::unordered_map<std::string, std::vector<std::string>> aux_var_set;
  std::vector<ir::Node *> adam_ops;
  for (ir::Node *node : result.Nodes()) {
    if (node->IsOp()) {
      GetSpecifiedOpsAndVars(fuse_op, aux_var_names, node, &adam_ops,
                             &aux_var_set);
    }
  }

  PADDLE_ENFORCE_NE(adam_ops.size(), 0, "Not found %s", fuse_op);
  VLOG(10) << "Find adam: " << adam_ops.size();

  // Sort the parameters
  SortVarsName(aux_var_names[0], &aux_var_set, &adam_ops);

  // Fuse the space of Gradient
  // Fuse Scale Op
  //  FuseAdamOps(aux_var_set,adam_ops, &result);
  FuseScaleOps(aux_var_set.at("Beta1Pow"), adam_ops, &result);
  FuseScaleOps(aux_var_set.at("Beta2Pow"), adam_ops, &result);

  // Fuse the space of "Moment1", "Moment2", "Beta1Pow", "Beta2Pow"
  // alloc_continuous_space
  if (!result.Has(kRunOnlyOnceProgram)) {
    result.Set(kRunOnlyOnceProgram, new RunOnlyOnceProgram);
  }
  result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).emplace_back();
  auto &program_desc =
      result.Get<RunOnlyOnceProgram>(kRunOnlyOnceProgram).back();
  auto *global_block = program_desc.MutableBlock(0);

  AppendAllocContinuousSpace(aux_var_set.at("Beta1Pow"), true /*copy_data*/,
                             global_block);
  AppendAllocContinuousSpace(aux_var_set.at("Beta2Pow"), true /*copy_data*/,
                             global_block);
  AppendAllocContinuousSpace(aux_var_set.at("Moment1"), true /*copy_data*/,
                             global_block);
  AppendAllocContinuousSpace(aux_var_set.at("Moment2"), true /*copy_data*/,
                             global_block);

  VLOG(10) << "FuseAdamOpPass Over ";
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

void FuseAdamOpPass::FuseScaleOps(const std::vector<std::string> &beta_1_pow,
                                  const std::vector<ir::Node *> &adam_ops,
                                  ir::Graph *graph) const {
  // Collect scale_ops
  const std::string scale_op_name = "scale";
  std::vector<ir::Node *> scale_ops;
  scale_ops.reserve(beta_1_pow.size());

  for (size_t i = 0; i < adam_ops.size(); ++i) {
    auto &beta_1_pow_name = beta_1_pow[i];
    auto beta_pow_iter = std::find_if(
        adam_ops[i]->inputs.begin(), adam_ops[i]->inputs.end(),
        [&beta_1_pow, &beta_1_pow_name](ir::Node *var_node) -> bool {
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

  int op_role = boost::get<int>(
      scale_ops[0]->Op()->GetAttr(OpProtoAndCheckerMaker::OpRoleAttrName()));

  // Check attributions
  PADDLE_ENFORCE_EQ(scale_ops.size(), beta_1_pow.size());
  float scale = boost::get<float>(scale_ops[0]->Op()->GetAttr("scale"));
  float bias = boost::get<float>(scale_ops[0]->Op()->GetAttr("bias"));
  bool bias_after_scale =
      boost::get<bool>(scale_ops[0]->Op()->GetAttr("bias_after_scale"));
  // check the scale's attr
  for (auto &scale_op : scale_ops) {
    PADDLE_ENFORCE_EQ(scale,
                      boost::get<float>(scale_op->Op()->GetAttr("scale")));
    PADDLE_ENFORCE_EQ(bias, boost::get<float>(scale_op->Op()->GetAttr("bias")));
    PADDLE_ENFORCE_EQ(
        bias_after_scale,
        boost::get<bool>(scale_op->Op()->GetAttr("bias_after_scale")));
  }

  // Add reset_dim, only fuse the scale ops
  int num_scale_op = static_cast<int>(beta_1_pow.size());
  OpDesc reset_dim_desc1;
  reset_dim_desc1.SetType("reset_dim");
  reset_dim_desc1.SetInput("Input", {beta_1_pow[0]});
  reset_dim_desc1.SetOutput("Output", {beta_1_pow[0]});
  reset_dim_desc1.SetAttr("new_dim", std::vector<int>{num_scale_op});
  reset_dim_desc1.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
  VLOG(10) << "Insert reset_dim to graph";
  // Insert to graph
  auto reset_dim_node1 = graph->CreateOpNode(&reset_dim_desc1);
  for (auto scale_op : scale_ops) {
    reset_dim_node1->inputs.insert(reset_dim_node1->inputs.begin(),
                                   scale_op->inputs.begin(),
                                   scale_op->inputs.end());
    for (auto &input : scale_op->inputs) {
      std::replace(input->outputs.begin(), input->outputs.end(), scale_op,
                   reset_dim_node1);
    }
  }

  // Add fused scale
  OpDesc scale_desc;
  scale_desc.SetType("scale");
  scale_desc.SetInput("X", {beta_1_pow[0]});
  scale_desc.SetOutput("Out", {beta_1_pow[0]});
  scale_desc.SetAttr("scale", scale);
  scale_desc.SetAttr("bias", bias);
  scale_desc.SetAttr("bias_after_scale", bias_after_scale);
  // Op should not have op_role, but it is used by ParallelExecutor.
  scale_desc.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
  auto scale_node = graph->CreateOpNode(&scale_desc);
  ir::Node *dep_var = graph->CreateControlDepVar();
  reset_dim_node1->outputs.emplace_back(dep_var);
  dep_var->inputs.emplace_back(reset_dim_node1);
  scale_node->inputs.emplace_back(dep_var);
  dep_var->outputs.emplace_back(scale_node);

  // Reset dims
  OpDesc reset_dim_desc2;
  reset_dim_desc2.SetType("reset_dim");
  reset_dim_desc2.SetInput("Input", {beta_1_pow[0]});
  reset_dim_desc2.SetOutput("Output", {beta_1_pow[0]});
  reset_dim_desc2.SetAttr("new_dim", std::vector<int>{1});
  reset_dim_desc2.SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(), op_role);
  auto reset_dim_node2 = graph->CreateOpNode(&reset_dim_desc2);
  auto dep_var2 = graph->CreateControlDepVar();

  scale_node->outputs.emplace_back(dep_var2);
  dep_var2->inputs.emplace_back(scale_node);

  reset_dim_node2->inputs.emplace_back(dep_var2);
  dep_var2->outputs.emplace_back(reset_dim_node2);

  for (auto &scale_op : scale_ops) {
    reset_dim_node2->outputs.insert(reset_dim_node2->outputs.begin(),
                                    scale_op->outputs.begin(),
                                    scale_op->outputs.end());
    for (auto &output : scale_op->outputs) {
      std::replace(output->inputs.begin(), output->inputs.end(), scale_op,
                   reset_dim_node2);
    }
  }

  // Delete scale_op
  for (auto &scale_op : scale_ops) {
    graph->RemoveNode(scale_op);
  }
}

void FuseAdamOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &args, bool copy_data,
    BlockDesc *global_block) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_continuous_space");
  op_desc->SetInput("Input", args);
  op_desc->SetOutput("Output", args);
  op_desc->SetAttr("copy_data", copy_data);
  //  op_desc->SetAttr("constant", static_cast<float >(0));
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fuse_adam_op_pass, paddle::framework::details::FuseAdamOpPass);
