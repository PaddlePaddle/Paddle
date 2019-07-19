// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <map>
#include <queue>
#include <string>
#include <unordered_set>
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_optimize_pass.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/op_info.h"

// NOTE(dzhwinter): inplace means one op output variable reuse the input space.
// By our design, one operator only can read its input(const Variable),
// write its output(non-const Variable). If one operator is inplaced, means
// user have chance to write the space before reading happens.
// Especially when some optimize code writing style is applied.
//
//
// /* wrong case in operator */
// /*In this case, a larger allocation is allocated, input content is lost*/
// const Tensor* in = ctx.Input<Tensor>("In")
// Tensor* out = ctx.Output<Tensor>("Out");
// auto* out_ptr = out->mutable_data<T>(ctx.GetPlace());
// out_ptr[0] = 0;  // input contect is overwrited.

// NOTE(dzhwinter):
// Only for backward compacity and stable. if enable_inplace_whitelist is turn
// on.
// only the ops in whitelist will be use inplace strategy.
// if not, all the op will be inplaced if it registered with InplaceClass
DEFINE_bool(
    enable_inplace_whitelist, false,
    "If this option turns on, only these op in whitelist can be inplaced."
    "If it turns off, all of the running op can be candidate of inplaced op."
    "Such as scale, elementwise_add"
    "By default, it's turned off");

namespace paddle {
namespace framework {
namespace ir {

// clang-format off
const std::string kInplacedOpWhiteList[] = { // NOLINT
    "sigmoid",
    "exp",
    "relu",
    "tanh",
    "sqrt",
    "ceil",
    "floor",
    "reciprocal",
    "relu6",
    "soft_relu",
    "hard_sigmoid",
    "batch_norm",
    "batch_norm_grad",
    "sum",
    "sum_grad",
    "scale",
    "reshape",
    "elementwise_add",
    "elementwise_add_grad",
};

// FIXME(zjl): Shapes of in-out of some ops are exactly the same,
// but the static size during compiling time would be wrong.
// Use a flag to indicate such ops. Please fix me when found a better way.
static const std::unordered_set<std::string> kSameShapeOpWhiteSet{ // NOLINT
    "reshape2", "reshape2_grad"
};
// clang-format on

class InplacePass : public ir::Pass {
 public:
  InplacePass();

 protected:
  void ApplyImpl(ir::Graph *graph) const override;

 private:
  // Collect vars that cannot be reused
  // e.g.: subblock ops in/out, distributed ops in/out, op_role_var
  void CollectSkipVars(ir::Graph *graph,
                       const std::vector<ir::Node *> &ops) const;

  // Check whether var_name should be skipped
  bool IsSkipVar(const std::string &var_name) const;

  // Rename out with name of in, and guarantee that the graph is
  // still a SSA graph
  void RenameInOut(ir::Node *op, ir::Node *in, ir::Node *out) const;

  // Check whether var is the last version one in SSA graph
  bool IsLastVersionVar(ir::Node *var) const;

  // Check whether var is the first version one in SSA graph
  bool IsFirstVersionVar(ir::Node *var) const;

  // Check whether all `ops` is the preceding ops of `op`
  bool CheckOpDeps(ir::Node *op, const std::vector<ir::Node *> &ops) const;

  // Find nodes whose names are equal to the given name
  static std::unordered_set<ir::Node *> FindNodesByName(
      const std::string &name, const std::vector<ir::Node *> &nodes);

  // Collect inputs and outputs of op_desc
  static void CollectInputArgsOfOpDesc(
      const OpDesc *op_desc, std::unordered_multiset<std::string> *in_args);

  // Get all versions vars named var_name
  std::vector<ir::Node *> *AllVersionVars(const std::string &var_name) const;

 private:
  // SSA graph. var_name -> each version of vars
  mutable std::map<std::string, std::vector<ir::Node *>> ssa_map_;

  // Skip vars, including subblock ops in/out, distributed ops in/out,
  // op_role_var
  mutable std::unordered_set<std::string> skip_vars_;

  // Op whitelist which should not peform inplace
  // Only enabled when FLAGS_enable_inplace_whitelist is true.
  mutable std::unordered_set<std::string> whitelist_ops_;
};

InplacePass::InplacePass() {
  if (FLAGS_enable_inplace_whitelist) {
    for (auto &s : kInplacedOpWhiteList) {
      whitelist_ops_.emplace(s);
    }
  }
}

std::vector<ir::Node *> *InplacePass::AllVersionVars(
    const std::string &var_name) const {
  auto iter = ssa_map_.find(var_name);
  PADDLE_ENFORCE(iter != ssa_map_.end(), "cannot find var %s in ssa graph",
                 var_name);
  PADDLE_ENFORCE(!iter->second.empty(), "var %s is empty in ssa graph",
                 var_name);
  return &(iter->second);
}

bool InplacePass::IsSkipVar(const std::string &var_name) const {
  return skip_vars_.count(var_name) > 0;
}

bool InplacePass::IsFirstVersionVar(ir::Node *var) const {
  return AllVersionVars(var->Name())->front() == var;
}

bool InplacePass::IsLastVersionVar(ir::Node *var) const {
  return AllVersionVars(var->Name())->back() == var;
}

bool InplacePass::CheckOpDeps(ir::Node *op,
                              const std::vector<ir::Node *> &ops) const {
  std::unordered_set<ir::Node *> other_ops(ops.begin(), ops.end());
  other_ops.erase(op);
  if (other_ops.empty()) return true;

  // Traverse all preceding ops of op
  std::queue<ir::Node *> queue;
  std::unordered_set<ir::Node *> visited_ops;
  queue.push(op);
  visited_ops.insert(op);

  // Visit all preceding ops of `op`, and erase it from other_ops if it is
  // inside other_ops. Return true only if other_ops is empty(), which means
  // that all `ops` are preceding ops of `op`.
  while (!queue.empty()) {
    auto *cur_op = queue.front();
    queue.pop();

    for (auto *in_var : cur_op->inputs) {
      for (auto *in_op : in_var->inputs) {
        if (visited_ops.count(in_op) != 0) {
          continue;
        }

        visited_ops.insert(in_op);
        queue.push(in_op);
        other_ops.erase(in_op);
        if (other_ops.empty()) return true;
      }
    }
  }
  return false;
}

void InplacePass::CollectSkipVars(ir::Graph *graph,
                                  const std::vector<ir::Node *> &ops) const {
  // 1. Collect op role vars
  PADDLE_ENFORCE(graph->Has(kMemOptSkipVars), "Graph should have attr %s",
                 kMemOptSkipVars);
  auto &mem_opt_whitelist = graph->Get<MemOptSkipVars>(kMemOptSkipVars);
  for (const auto &var : mem_opt_whitelist) {
    skip_vars_.emplace(var);
  }
}

void InplacePass::RenameInOut(ir::Node *op, ir::Node *in_var,
                              ir::Node *out_var) const {
  auto out_var_name = out_var->Name();
  auto in_var_name = in_var->Name();

  auto &all_out_nodes = *AllVersionVars(out_var_name);
  auto &all_in_nodes = *AllVersionVars(in_var_name);

  auto iter = std::find(all_out_nodes.begin(), all_out_nodes.end(), out_var);
  PADDLE_ENFORCE(iter != all_out_nodes.end(), "Cannot find out var %s",
                 out_var_name);

  // The following codes are designed to guarantee that ssa_map_ is still
  // an ssa graph after inplace is performed.
  // Step 1: Rename the following versions of out_var as the name of in_var
  // Step 2: Remove the following versions of out_var and append them to in_var
  // Be careful that the inputs of input op of out_var should not be renamed,
  // but outputs should be renamed.
  auto original_iter = iter;
  while (iter != all_out_nodes.end()) {
    auto *node = *iter;
    /* Step 1 */
    node->RenameVar(in_var_name);
    if (iter != original_iter) {
      for (auto *in : node->inputs) {
        if (in->IsOp() && in->Op()) {
          in->Op()->RenameOutput(out_var_name, in_var_name);
          in->Op()->RenameInput(out_var_name, in_var_name);
          in->Op()->Flush();
        }
      }
    }

    for (auto *out : node->outputs) {
      if (out->IsOp() && out->Op()) {
        out->Op()->RenameOutput(out_var_name, in_var_name);
        out->Op()->RenameInput(out_var_name, in_var_name);
        out->Op()->Flush();
      }
    }

    /* Step 2 */
    all_in_nodes.emplace_back(node);
    ++iter;
  }

  /* Step 2 */
  all_out_nodes.erase(original_iter, all_out_nodes.end());

  if (all_out_nodes.empty()) {
    ssa_map_.erase(out_var_name);
  }
  op->Op()->RenameOutput(out_var_name, in_var_name);
  op->Op()->Flush();
}

std::unordered_set<ir::Node *> InplacePass::FindNodesByName(
    const std::string &name, const std::vector<ir::Node *> &nodes) {
  std::unordered_set<ir::Node *> ret;
  for (auto *node : nodes) {
    if (node->Name() == name) {
      ret.insert(node);
    }
  }
  return ret;
}

void InplacePass::CollectInputArgsOfOpDesc(
    const OpDesc *op_desc, std::unordered_multiset<std::string> *in_args) {
  in_args->clear();
  for (auto &in_name : op_desc->InputArgumentNames()) {
    in_args->insert(in_name);
  }
}

void InplacePass::ApplyImpl(ir::Graph *graph) const {
  // Step 1: topo sort ops, collect skip vars
  auto ops = ir::TopologySortOperations(*graph);
  CollectSkipVars(graph, ops);

  // Step 2: build ssa var map
  for (auto *op_node : ops) {
    for (auto *in : op_node->inputs) {
      PADDLE_ENFORCE(in->IsVar());
      // Only create a new var node when var first occurs in input of op.
      if (ssa_map_.count(in->Name()) == 0) {
        ssa_map_[in->Name()].emplace_back(in);
      }
    }

    // Always create a new var node for each output of op.
    for (auto *out : op_node->outputs) {
      PADDLE_ENFORCE(out->IsVar());
      ssa_map_[out->Name()].emplace_back(out);
    }
  }

  // Step 3: traverse ops and try inplace if possible
  bool use_cuda = Get<bool>(kUseCuda);
  VLOG(4) << "Inplace pass is applied when use_cuda = "
          << (use_cuda ? "true" : "false");

  for (auto *op_node : ops) {
    PADDLE_ENFORCE_NOT_NULL(op_node->Op(), "op_desc is nullptr");

    auto *op_desc = op_node->Op();
    auto op_type = op_desc->Type();

    // Skip op inside whitelist
    if (whitelist_ops_.count(op_type) > 0) {
      continue;
    }

    auto &infer_inplace = OpInfoMap::Instance().Get(op_type).infer_inplace_;

    if (!infer_inplace) {
      continue;
    }

    auto in_to_outs = infer_inplace(*op_desc, use_cuda);
    if (in_to_outs.empty()) continue;

    std::unordered_multiset<std::string> all_in_args;
    CollectInputArgsOfOpDesc(op_desc, &all_in_args);

    for (auto &pair : in_to_outs) {
      auto &in_param = pair.first;
      auto &out_param = pair.second;

      auto &in_args = op_desc->Input(in_param);
      auto &out_args = op_desc->Output(out_param);

      if (in_args.empty()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param
                << ") is empty in " << op_type;
        continue;
      }

      if (out_args.empty()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ") is empty in " << op_type;
        continue;
      }

      auto &in_arg = in_args[0];
      auto &out_arg = out_args[0];

      if (IsSkipVar(in_arg)) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is skipped in " << op_type;
        continue;
      }

      if (IsSkipVar(out_arg)) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " is skipped in " << op_type;
        continue;
      }

      if (in_arg == out_arg) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is the same with Output(" << out_param << ")=" << out_arg
                << " in " << op_type;
        continue;
      }

      size_t in_arg_occur_times = all_in_args.count(in_arg);
      if (in_arg_occur_times > 1) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " occurs " << in_arg_occur_times << " times in input of op "
                << op_type;
        continue;
      }

      auto in_nodes = FindNodesByName(in_arg, op_node->inputs);
      PADDLE_ENFORCE(!in_nodes.empty(), "Input(%s)=%s cannot be found in op %s",
                     in_param, in_arg, op_type);

      if (in_nodes.size() > 1) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " occurs in other inputs of " << op_type;
        continue;
      }

      auto *in_node = *in_nodes.begin();

      if (!NodeCanReused(in_node)) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is not reusable in " << op_type;
        continue;
      }

      if (!IsLastVersionVar(in_node)) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is not the last version in " << op_type;
        continue;
      }

      // If in_node is used as inputs of many ops, check whether all of that ops
      // depends on op_node. If not, in_node cannot be inplaced.
      if (in_node->outputs.size() > 1 &&
          !CheckOpDeps(op_node, in_node->outputs)) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is not lastly used in " << op_type;
        continue;
      }

      auto out_nodes = FindNodesByName(out_arg, op_node->outputs);
      PADDLE_ENFORCE(!out_nodes.empty(),
                     "Output(%s)=%s cannot be found in op %s", out_param,
                     out_arg, op_type);

      PADDLE_ENFORCE_EQ(
          out_nodes.size(), 1,
          "Wrong graph: Output(%s)=%s occurs in other outputs of op %s",
          out_param, out_arg, op_type);

      if (!FindNodesByName(in_arg, op_node->outputs).empty()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " occurs in output of op " << op_type;
        continue;
      }

      if (!FindNodesByName(out_arg, op_node->inputs).empty()) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " occurs in input of op " << op_type;
        continue;
      }

      auto *out_node = *out_nodes.begin();

      if (!IsFirstVersionVar(out_node)) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " does not occur first in op " << op_type;
        continue;
      }

      if (!NodeCanReused(out_node)) {
        VLOG(4) << "Cannot inplace because Output(" << out_param
                << ")=" << out_arg << " is not reusable in " << op_type;
        continue;
      }

      if (in_node->Var()->GetType() != out_node->Var()->GetType()) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is not the same type with "
                << "Output(" << out_param << ")=" << out_arg << " in "
                << op_type;
        continue;
      }

      if (NodeSize(*in_node->Var()) != NodeSize(*out_node->Var()) &&
          kSameShapeOpWhiteSet.count(op_desc->Type()) == 0) {
        VLOG(4) << "Cannot inplace because Input(" << in_param << ")=" << in_arg
                << " is not the same size with "
                << "Output(" << out_param << ")=" << out_arg << " in "
                << op_type;
        continue;
      }

      VLOG(4) << "Rename " << out_node->Name() << " with " << in_node->Name()
              << " in " << op_type;
      RenameInOut(op_node, in_node, out_node);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_pass, paddle::framework::ir::InplacePass)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
