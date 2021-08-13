// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>
#include <unordered_map>

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/memory_optimize_pass/memory_reuse_pass.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace details {
class ComputationOpHandle;
struct VarHandle;
}  // namespace details
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Graph;

class InplaceAddToOpPass : public MemoryReusePass {
 protected:
  std::string ReuseType() const override { return "inplace_addto"; }

  void Run(Graph *graph) const override;

  void ApplyImpl(ProgramDesc *main_program,
                 ProgramDesc *startup_program) const override;

 private:
  // 1. Add last living op of in_var, add any last living op of out_var
  // 2. Set reference count of in_var to be 2
  void UpdateLastLiveOpOfVar(details::ComputationOpHandle *op,
                             details::VarHandle *in_var,
                             details::VarHandle *out_var) const override {
    size_t scope_idx = op->GetScopeIdx();
    auto *last_live_ops_of_vars_ =
        &Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);
    auto *var_infos_ = &(Get<MemOptVarInfoMapList>(kMemOptVarInfoMapList));
    auto out_var_op_iter =
        (*last_live_ops_of_vars_)[scope_idx].find(out_var->Name());

    // In Reduce mode, some output variable(gradient of parameter) does not have
    // last live ops
    details::ComputationOpHandle *last_live_op_of_in_var = nullptr;
    if (out_var_op_iter == (*last_live_ops_of_vars_)[scope_idx].end()) {
      last_live_op_of_in_var = op;
    } else {
      PADDLE_ENFORCE_EQ(
          out_var_op_iter->second.ops().empty(), false,
          platform::errors::InvalidArgument(
              "Var(%s)'s last live op should not empty.", out_var->Name()));
      last_live_op_of_in_var = *(out_var_op_iter->second.ops().begin());
    }

    auto *last_live_ops_of_in_var =
        (*last_live_ops_of_vars_)[scope_idx][in_var->Name()].mutable_ops();
    // last_live_ops_of_in_var->clear();
    last_live_ops_of_in_var->insert(last_live_op_of_in_var);

    auto in_var_info_iter = (*var_infos_)[scope_idx].find(in_var->Name());
    PADDLE_ENFORCE_NE(
        in_var_info_iter, (*var_infos_)[scope_idx].end(),
        platform::errors::NotFound("Cannot find variable %s.", in_var->Name()));

    in_var_info_iter->second->SetRefCnt(2);  // before inplace, it is 1
  }
};

void InplaceAddToOpPass::Run(Graph *graph) const {
  const auto &last_live_ops =
      Get<std::vector<LastLiveOpsOfVars>>(kLastLiveOpsOfVars);

  bool use_cuda = Get<bool>(kUseCuda);

  // Currently, only perform InplaceAddToOpPass on cuda place
  if (!use_cuda) {
    return;
  }

  // Step 1: Build a reverse map of last_live_ops
  // i.e.: op -> vars
  std::unordered_map<details::ComputationOpHandle *,
                     std::unordered_map<std::string, ir::Node *>>
      candidate_ops;
  for (auto &each_scope_ops : last_live_ops) {
    for (auto &pair : each_scope_ops) {
      // If variable has more than 1 last lived ops, this variable cannot
      // be inplaced.
      if (pair.second.ops().size() != 1) {
        continue;
      }

      auto *op = *(pair.second.ops().begin());
      const std::string &op_type = op->GetOp()->Type();
      const framework::OpDesc *op_desc = op->Node()->Op();
      PADDLE_ENFORCE_NOT_NULL(
          op_desc, platform::errors::NotFound("Op(%s) can not find opdesc.",
                                              op->Name()));

      // only grad op should be processed.
      if (op_type != "grad_add") {
        continue;
      }

      const std::string &var_name = pair.first;
      auto in_nodes = this->FindNodesByName(var_name, op->Node()->inputs);
      if (in_nodes.size() == 1) {
        candidate_ops[op][var_name] = *in_nodes.begin();
      }
      VLOG(4) << "Find op " << op_type << " with input(" << var_name
              << ") that can do inplace add to";
    }
  }

  // Step 2: Check which vars can be inplaced indeed
  for (auto &op_vars_pair : candidate_ops) {
    auto *op = op_vars_pair.first;

    // The original gradient accumulation is g = sum(g_0, g_1,..., g_n), and it
    // could be changed as follws if inplace addto is enabled:
    // g_sum_0 = g_0
    // g_sum_1 = grad_add(g_sum_0, g_1)
    // g_sum_2 = grad_add(g_sum_1, g_2)
    // ...
    // g_sum_n = grad_add(g_sum_n-1, g_n)

    // here we will add inplace for each grad_add, for example, for the first
    // grad_add, g_sum_0 -> g1, g_sum_1 -> g1, and set grad_add as skipped.

    const std::string &op_type = op->GetOp()->Type();

    PADDLE_ENFORCE_EQ(op->Node()->inputs.size(), 2,
                      platform::errors::InvalidArgument(
                          "The size of inputs of %s should be 2, but got %d",
                          op_type, op->Node()->inputs.size()));

    PADDLE_ENFORCE_EQ(op->Node()->outputs.size(), 1,
                      platform::errors::InvalidArgument(
                          "The size of outputs of %s should be 1, but got %d",
                          op_type, op->Node()->outputs.size()));

    auto *left_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->inputs[0]->Wrapper<details::VarHandleBase>()));
    auto *right_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->inputs[1]->Wrapper<details::VarHandleBase>()));
    auto *out_var_ptr = dynamic_cast<details::VarHandle *>(
        &(op->Node()->outputs[0]->Wrapper<details::VarHandleBase>()));

    if (left_var_ptr == nullptr || right_var_ptr == nullptr ||
        out_var_ptr == nullptr) {
      continue;
    }

    // auto *left_generated_op = dynamic_cast<details::ComputationOpHandle *>(
    //     left_var_ptr->GeneratedOp());

    auto *right_generated_op = dynamic_cast<details::ComputationOpHandle *>(
        right_var_ptr->GeneratedOp());

    auto *out_generated_op = dynamic_cast<details::ComputationOpHandle *>(
        out_var_ptr->GeneratedOp());

    // NOTE(zhiqiu): currently, only conv2d_grad supports addto strategy
    if (right_generated_op->Name() != "conv2d_grad") {
      continue;
    }

    // NOTE(zhiqiu): Normally, if we inplace a->b, we should let a generated
    // before b. However, in the situation of inplace addto, we do not care
    // the order, since a+b is equal to b+a. Is there any exception for that?

    // AddDependencyVar(right_generated_op, left_generated_op);
    // no need, as discussed above.

    // step (a): inplace right_var->left_var of grad_add

    this->AddReuseVar(right_generated_op, left_var_ptr, right_var_ptr);
    UpdateLastLiveOpOfVar(right_generated_op, left_var_ptr, right_var_ptr);
    VLOG(4) << "Inplace performed in op " << right_generated_op->GetOp()->Type()
            << ": " << left_var_ptr->Name() << " -> " << right_var_ptr->Name()
            << ". Debug String is: "
            << right_generated_op->GetOp()->DebugString()
            << ". ReuseType: " << ReuseType();

    // step (b): inplace out -> right_var of grad_add

    this->AddReuseVar(out_generated_op, right_var_ptr, out_var_ptr, true);

    VLOG(4) << "Inplace performed in op " << op_type << ": "
            << left_var_ptr->Name() << " -> " << out_var_ptr->Name()
            << ". Debug String is: " << op->GetOp()->DebugString()
            << ". ReuseType: " << ReuseType();

    // step (c): make right_var cannot inplace afterwards. canbe done
    // aotomatically since CollectReusedVars is called before any reuse.

    // step (d): make right_var's generated op use addto
    right_generated_op->GetOp()->SetAttr("use_addto", true);

    // step (e): make grad_add skip running
    op->SetSkipRunning(true);
  }
}

static bool IsValidConv2DGradDataGradNode(const Node &node) {
  if (node.inputs.empty()) return false;
  auto *generated_op = node.inputs[0];
  auto *op_desc = generated_op->Op();
  if (op_desc == nullptr || op_desc->Type() != "conv2d_grad") {
    return false;
  }
  const auto &outputs = op_desc->Outputs();
  auto iter = outputs.find(GradVarName("Input"));
  return iter != outputs.end() && !iter->second.empty() &&
         iter->second[0] == node.Name() &&
         !op_desc->GetAttrIfExists<bool>("use_addto");
}

static bool IsDownstreamNode(const Node &upstream, const Node &downstream) {
  std::queue<const Node *> q;
  std::unordered_set<const Node *> visited;
  q.push(&upstream);
  visited.insert(&upstream);
  while (!q.empty()) {
    const auto *cur = q.front();
    q.pop();
    if (cur == &downstream) {
      return true;
    }

    for (const auto *out : cur->outputs) {
      if (visited.count(out) == 0) {
        visited.insert(out);
        q.push(out);
      }
    }
  }
  return false;
}

static void BuildInplaceAddToGraph(Node *in_var_0, Node *in_var_1,
                                   Node *out_var, Graph *graph) {
  auto *grad_add_op = out_var->inputs[0];

  // Cut the connection between in_var_0 and grad_add_op
  in_var_0->outputs.erase(std::remove(in_var_0->outputs.begin(),
                                      in_var_0->outputs.end(), grad_add_op),
                          in_var_0->outputs.end());
  grad_add_op->inputs.erase(std::remove(grad_add_op->inputs.begin(),
                                        grad_add_op->inputs.end(), in_var_0),
                            grad_add_op->inputs.end());

  // Replace grad_add_op with share_buffer op
  auto *grad_add_op_desc = grad_add_op->Op();
  grad_add_op_desc->SetType("share_buffer");
  grad_add_op_desc->SetInput("X", {in_var_1->Name()});
  grad_add_op_desc->SetOutput("Out", {out_var->Name()});
  grad_add_op_desc->SetOutput("XOut", {in_var_1->Name()});
  grad_add_op_desc->SetAttr("share_dims", std::vector<bool>(1, true));

  // Add share_buffer op between in_var_0 and in_var_1
  OpDesc share_buffer_op;
  share_buffer_op.SetType("share_buffer");
  share_buffer_op.SetInput("X", {in_var_0->Name()});
  share_buffer_op.SetOutput("Out", {in_var_1->Name()});
  share_buffer_op.SetOutput("XOut", {in_var_0->Name()});
  share_buffer_op.SetAttr("share_dims", std::vector<bool>(1, false));

  auto *new_share_buffer_op = graph->CreateOpNode(&share_buffer_op);
  new_share_buffer_op->inputs.push_back(in_var_0);
  in_var_0->outputs.push_back(new_share_buffer_op);
  new_share_buffer_op->outputs.push_back(in_var_1);
  in_var_1->inputs.push_back(new_share_buffer_op);

  auto *dep_var = graph->CreateControlDepVar();
  new_share_buffer_op->outputs.push_back(dep_var);
  dep_var->inputs.push_back(new_share_buffer_op);

  auto in_var_1_gen_op = in_var_1->inputs[0];
  in_var_1_gen_op->inputs.push_back(dep_var);
  dep_var->outputs.push_back(in_var_1_gen_op);

  in_var_1_gen_op->Op()->SetAttr("use_addto", true);
}

static std::unordered_map<std::string, std::vector<Node *>>
GetAllVersionVarsMap(const Graph &graph) {
  const auto &nodes = graph.Nodes();
  std::unordered_map<Node *, size_t> deps;
  std::vector<Node *> sorted_nodes;
  sorted_nodes.reserve(nodes.size());

  std::queue<Node *> q;
  for (auto *node : nodes) {
    size_t in_degree = node->inputs.size();
    if (in_degree == 0) {
      q.push(node);
      sorted_nodes.push_back(node);
    } else {
      deps[node] = node->inputs.size();
    }
  }

  while (!q.empty()) {
    auto *cur = q.front();
    q.pop();
    for (auto *node : cur->outputs) {
      if (--deps.at(node) == 0) {
        sorted_nodes.push_back(node);
        q.push(node);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      sorted_nodes.size(), nodes.size(),
      platform::errors::PermissionDenied("Wrong toplogical sort algorithm."));
  std::unordered_map<std::string, std::vector<Node *>> result;
  for (auto *node : sorted_nodes) {
    if (node->IsVar() && !node->IsCtrlVar()) {
      result[node->Name()].push_back(node);
    }
  }
  return result;
}

void InplaceAddToOpPass::ApplyImpl(ProgramDesc *main_program,
                                   ProgramDesc *startup_program) const {
  if (!Get<bool>(kUseCuda)) return;

  Graph graph(*main_program);
  auto all_ver_vars = GetAllVersionVarsMap(graph);

  const auto all_nodes = graph.Nodes();  // Deep copy
  std::unordered_set<std::string> reused_in_vars;
  std::unordered_set<std::string> reused_out_vars;
  for (auto *node : all_nodes) {
    if (!node->IsOp() || node->Op() == nullptr ||
        node->Op()->Type() != "grad_add") {
      continue;
    }

    VLOG(10) << "Found grad_add op";

    // Step 1: find input vars first
    std::vector<Node *> input_vars;
    input_vars.reserve(2);
    for (auto *in : node->inputs) {
      if (in->IsCtrlVar() || in->Name() == kEmptyVarName) {
        continue;
      }
      PADDLE_ENFORCE_LT(input_vars.size(), 2,
                        platform::errors::InvalidArgument(
                            "The size of inputs of grad_add should be 2."));
      input_vars.push_back(in);
    }

    if (input_vars.size() != 2) {  // may have kEmptyVarName
      continue;
    }

    bool is_first_var_valid = IsValidConv2DGradDataGradNode(*input_vars[0]);
    bool is_second_var_valid = IsValidConv2DGradDataGradNode(*input_vars[1]);
    if (!is_first_var_valid && !is_second_var_valid) {
      continue;
    }

    VLOG(10) << "validation " << is_first_var_valid << " "
             << is_second_var_valid;

    // make sure that input_vars[1] is always the Input@GRAD of conv2d_grad op
    if (is_first_var_valid) {
      std::swap(input_vars[0], input_vars[1]);
    }

    // Step 2: find the unique output var
    Node *output_var = nullptr;
    std::string output_var_name = node->Op()->Output("Out")[0];
    PADDLE_ENFORCE_NE(output_var_name, kEmptyVarName,
                      platform::errors::InvalidArgument(
                          "Output of grad_add should be provided."));
    for (auto *out : node->outputs) {
      if (output_var_name == out->Name()) {
        output_var = out;
        break;
      }
    }
    PADDLE_ENFORCE_NOT_NULL(output_var,
                            platform::errors::InvalidArgument(
                                "Output of grad_add should be provided."));

    VLOG(10) << "Check inplace chain: " << input_vars[0]->Name() << " -> "
             << input_vars[1]->Name() << " -> " << output_var->Name();

    // Step 3: check whether input_vars[0]->generated_op is not the downstream
    // op of input_vars[0]->generated_op. If yes, circle would occur.
    if (!input_vars[0]->inputs.empty() && !input_vars[1]->inputs.empty()) {
      auto *gen_op_0 = input_vars[0]->inputs[0];
      auto *gen_op_1 = input_vars[1]->inputs[0];
      if (IsDownstreamNode(*gen_op_1, *gen_op_0)) {
        VLOG(10) << "Downstream node detected, cannot inplace addto";
        continue;
      }
    }

    // Step 4: name not the same
    if (input_vars[0]->Name() == input_vars[1]->Name() ||
        input_vars[0]->Name() == output_var->Name() ||
        input_vars[1]->Name() == output_var->Name()) {
      continue;
    }

    // Step 5: check var version. The inplace var chain is: input_vars[0] ->
    // input_vars[1] -> output_var
    // Therefore, input_vars[0] must be last version, input_vars[1] must be 1st
    // version and last version, and output_var must be the 1st version.
    auto iter = all_ver_vars.find(input_vars[0]->Name());
    PADDLE_ENFORCE_EQ(iter != all_ver_vars.end(), true,
                      platform::errors::InvalidArgument(
                          "Variable %s not found.", input_vars[0]->Name()));
    if (iter->second[iter->second.size() - 1] != input_vars[0]) continue;

    iter = all_ver_vars.find(input_vars[1]->Name());
    if (iter->second.size() != 1) continue;
    PADDLE_ENFORCE_EQ(iter->second[0], input_vars[1],
                      platform::errors::InvalidArgument(
                          "Variable %s not found.", input_vars[1]->Name()));
    iter = all_ver_vars.find(output_var->Name());
    if (iter->second[0] != output_var) continue;

    // Step 6: input_vars[0] and input_vars[1] should only have one output op!
    // This output op must be grad_add op.
    if (input_vars[0]->outputs.size() != 1 ||
        input_vars[1]->outputs.size() != 1) {
      continue;
    }

    // Step 7: check whether the var has been reused
    if (reused_in_vars.count(input_vars[0]->Name()) > 0 ||
        reused_in_vars.count(input_vars[1]->Name()) > 0 ||
        reused_out_vars.count(input_vars[1]->Name()) > 0 ||
        reused_out_vars.count(output_var->Name()) > 0) {
      continue;
    }

    VLOG(10) << "inplace occurs at " << input_vars[0]->Name() << " -> "
             << input_vars[1]->Name() << " -> " << output_var->Name();
    // Step 8: inplace addto can be performed now!
    BuildInplaceAddToGraph(input_vars[0], input_vars[1], output_var, &graph);
    reused_in_vars.insert(input_vars[0]->Name());
    reused_in_vars.insert(input_vars[1]->Name());
    reused_out_vars.insert(input_vars[1]->Name());
    reused_out_vars.insert(output_var->Name());
  }

  // Convert Graph to main_program
  ProgramDesc tmp;
  GraphToProgram(graph, &tmp);
  main_program->CopyFrom(*tmp.Proto());
  main_program->Flush();
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_addto_op_pass, paddle::framework::ir::InplaceAddToOpPass)
    .RequirePassAttr(paddle::framework::ir::kMemOptVarInfoMapList)
    .RequirePassAttr(paddle::framework::ir::kLastLiveOpsOfVars)
    .RequirePassAttr(paddle::framework::ir::kUseCuda);
