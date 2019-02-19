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

#include "paddle/fluid/framework/details/fuse_optimizer_op_pass.h"
#include <algorithm>
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace details {

std::unique_ptr<ir::Graph> FuseOptimizerOpPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  ir::Graph &result = *graph;

  const std::string fuse_op_type = GetOpType();
  const std::vector<std::string> aux_var_names = GetAuxiliaryVarNames();

  // Step 1: Get the specified op and auxiliary variables.
  std::vector<ir::Node *> topo_nodes = ir::TopologySortOperations(result);
  std::unordered_map<std::string, std::vector<std::string>> aux_var_set;
  std::vector<ir::Node *> opt_ops;
  for (auto &node : topo_nodes) {
    GetSpecifiedOpsAndVars(fuse_op_type, aux_var_names, node, &opt_ops,
                           &aux_var_set);
  }

  VLOG(10) << "Find " << fuse_op_type << " operators: " << opt_ops.size();
  if (opt_ops.size() == 0) {
    return std::move(graph);
  }

  if (result.Has(kFusedOptType)) {
    VLOG(10)
        << "Currently only support fusing one type optimizer op. Has fused "
        << result.Get<FusedOptType>(kFusedOptType);
    return std::move(graph);
  } else {
    result.Set(kFusedOptType, new FusedOptType);
  }
  result.Get<FusedOptType>(kFusedOptType) = fuse_op_type;

  // Step 2: Insert fused_var_name to FusedVars, and the FusedVars need be
  // initialized in scopes before execution.
  if (!result.Has(kFusedVars)) {
    result.Set(kFusedVars, new FusedVars);
  }

  std::unordered_map<std::string, std::string> fused_vars_name;
  fused_vars_name.reserve(aux_var_names.size() + 1);
  const std::string prefix(kFusedVarNamePrefix);
  for (auto &var_name : aux_var_names) {
    auto fused_var_name = prefix + "_" + fuse_op_type + "_" + var_name;
    fused_vars_name.emplace(var_name, fused_var_name);
    result.Get<FusedVars>(kFusedVars).emplace_back(fused_var_name);
  }

  // Step 3: Sort the parameters and auxiliary variables according
  // to parameters' name to make variables' name correspond correctly.
  PADDLE_ENFORCE(result.Has(kParamsAndGrads), "Does't find kParamsAndGrads.");
  auto &params_grads = result.Get<ParamsAndGrads>(kParamsAndGrads);
  PADDLE_ENFORCE_EQ(params_grads.size(), aux_var_set.begin()->second.size());
  SortVarsName(params_grads, &aux_var_set, &opt_ops);

  // Step 4: Alloc continuous space for Moment1, Moment2, Beta1Pow, Beta2Pow
  // of all the optimizer ops separately.
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

  // Step 6: Fuse optimizer Ops and Scale Ops
  FuseOptimizerOps(aux_var_set, fused_vars_name, opt_ops, &result);

  // Step 7: Remove optimizer Ops
  for (auto &opt_op : opt_ops) {
    graph->RemoveNode(opt_op);
  }

  return std::move(graph);
}

void FuseOptimizerOpPass::SortVarsName(
    const std::vector<std::pair<std::string, std::string>> &params_grads,
    std::unordered_map<std::string, std::vector<std::string>> *aux_vars_set,
    std::vector<ir::Node *> *ops) const {
  PADDLE_ENFORCE_NE(aux_vars_set->count("Param"), static_cast<size_t>(0));
  auto &param_vec = aux_vars_set->at("Param");

  std::vector<size_t> param_sort_idx;
  param_sort_idx.reserve(param_vec.size());
  for (size_t i = 0; i < param_vec.size(); ++i) {
    param_sort_idx.emplace_back(i);
  }
  std::sort(param_sort_idx.begin(), param_sort_idx.end(),
            [&param_vec](size_t a, size_t b) -> bool {
              return param_vec[a] < param_vec[b];
            });

  //  for (auto &p_g : params_grads) {
  //    auto iter = std::find(param_vec.begin(), param_vec.end(), p_g.first);
  //    PADDLE_ENFORCE(iter != param_vec.end());
  //    auto idx = std::distance(param_vec.begin(), iter);
  //    param_sort_idx.emplace_back(idx);
  //  }

  for (auto &aux_vars : *aux_vars_set) {
    std::vector<std::string> sorted_vars;
    sorted_vars.reserve(aux_vars.second.size());
    for (size_t i = 0; i < aux_vars.second.size(); ++i) {
      sorted_vars.emplace_back(aux_vars.second.at(param_sort_idx[i]));
    }
    std::swap(aux_vars.second, sorted_vars);

    std::stringstream out;
    for (auto &var_name : aux_vars.second) {
      out << var_name << " ";
    }
    VLOG(10) << aux_vars.first << ": " << out.str();
  }

  std::vector<ir::Node *> sorted_ops;
  sorted_ops.reserve(ops->size());
  for (size_t i = 0; i < ops->size(); ++i) {
    sorted_ops.emplace_back(ops->at(param_sort_idx[i]));
  }
  std::swap(*ops, sorted_ops);
}

void FuseOptimizerOpPass::GetSpecifiedOpsAndVars(
    const std::string &op_type, const std::vector<std::string> &aux_vars_name,
    ir::Node *node, std::vector<ir::Node *> *ops,
    std::unordered_map<std::string, std::vector<std::string>> *aux_args_name)
    const {
  if (node->Op()->Type() != op_type) return;

  for (auto &var_n : aux_vars_name) {
    auto arg_names = node->Op()->Input(var_n);
    PADDLE_ENFORCE_EQ(arg_names.size(), static_cast<size_t>(1));
    (*aux_args_name)[var_n].emplace_back(arg_names[0]);
  }
  ops->emplace_back(node);
}

void FuseOptimizerOpPass::AppendAllocContinuousSpace(
    const std::vector<std::string> &args, const std::string &out_arg,
    bool copy_data, BlockDesc *global_block) const {
  auto op_desc = global_block->AppendOp();
  op_desc->SetType("alloc_continuous_space");
  op_desc->SetInput("Input", args);
  op_desc->SetOutput("Output", args);
  op_desc->SetOutput("FusedOutput", {out_arg});
  op_desc->SetAttr("copy_data", copy_data);
}

void FuseOptimizerOpPass::InserInputAndOutputForOptOps(
    const std::vector<ir::Node *> &opt_ops, ir::Node *opt_node) const {
  for (auto opt_op : opt_ops) {
    // set inputs
    opt_node->inputs.insert(opt_node->inputs.begin(), opt_op->inputs.begin(),
                            opt_op->inputs.end());
    for (auto &input : opt_op->inputs) {
      replace(input->outputs.begin(), input->outputs.end(), opt_op, opt_node);
    }
    // set outputs
    opt_node->outputs.insert(opt_node->outputs.begin(), opt_op->outputs.begin(),
                             opt_op->outputs.end());
    for (auto &output : opt_op->outputs) {
      replace(output->inputs.begin(), output->inputs.end(), opt_op, opt_node);
    }
  }
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
