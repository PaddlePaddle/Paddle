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

#include "paddle/fluid/framework/details/inplace_op_pass.h"
#include <algorithm>
#include <deque>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/memory_optimize_pass.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
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

DECLARE_string(memory_optimize_debug);

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
// clang-format on

namespace paddle {
namespace framework {
namespace details {

static inline ir::Node* GetNextCascadeInplacedVar(ir::Node* var) {
  // if next op is inplaced, then return the output var
  // otherwise return nullptr
  PADDLE_ENFORCE(var && var->IsVar() && !var->IsCtrlVar());
  ir::Node* inplaced_var = nullptr;
  for (auto* next_op : var->outputs) {
    for (auto* output : next_op->outputs) {
      if (output->IsVar() && !output->IsCtrlVar() &&
          output->Name() == var->Name()) {
        inplaced_var = output;
      }
    }
  }
  return inplaced_var;
}

static inline ir::Node* GetPrevCascadeInplacedVar(ir::Node* var) {
  PADDLE_ENFORCE(var && var->IsVar() && !var->IsCtrlVar());
  if (var->inputs.empty()) return nullptr;
  auto* prev_op = var->inputs.at(0);
  auto input_it = std::find_if(prev_op->inputs.begin(), prev_op->inputs.end(),
                               [&](ir::Node* node) {
                                 if (node->IsVar() && !node->IsCtrlVar() &&
                                     node->Name() == var->Name()) {
                                   return true;
                                 } else {
                                   return false;
                                 }
                               });
  return input_it == prev_op->inputs.end() ? nullptr : *input_it;
}

InplacePass::InplacePass() : Pass() {
  if (FLAGS_enable_inplace_whitelist) {
    for (auto& s : kInplacedOpWhiteList) {
      whitelist_.emplace(s);
    }
  }
}

void InplacePass::InitSSAGraphNodes() const {
  std::unordered_map<std::string, std::unordered_set<ir::Node*>> all_vars;
  for (auto* op : view_.AllOps()) {
    for (auto* node : op->inputs) {
      if (!node->IsVar() || node->IsCtrlVar()) continue;
      if (all_vars[node->Name()].count(node) == 0) {
        all_vars[node->Name()].emplace(node);
        var_nodes_[node->Name()].emplace_back(node);
      }
    }
    for (auto* node : op->outputs) {
      if (!node->IsVar() || node->IsCtrlVar()) continue;
      if (all_vars[node->Name()].count(node) == 0) {
        all_vars[node->Name()].emplace(node);
        var_nodes_[node->Name()].emplace_back(node);
      }
    }
  }
}

void InplacePass::ApplyImpl(ir::Graph* graph) const {
  var_nodes_.clear();
  view_.Build(graph);
  InitSSAGraphNodes();

  auto cnt = 0;
  for (auto* op : view_.AllOps()) {
    VLOG(4) << "Handle op " << cnt++ << ": " << op->Name();
    if (FLAGS_enable_inplace_whitelist && !whitelist_.count(op->Name()))
      continue;
    TryInplaceOpInputOutput(op, graph);
  }
}

void InplacePass::InplaceModifyDesc(const std::string& var,
                                    const std::string& cache_var,
                                    const size_t& idx) const {
  for (size_t i = idx; i < view_.AllOps().size(); ++i) {
    ir::Node* op = view_.AllOps()[i];
    PADDLE_ENFORCE(op->IsOp() && op->Op());
    auto* op_desc = op->Op();
    op_desc->RenameInput(var, cache_var);
    op_desc->RenameOutput(var, cache_var);

    op_desc->Flush();
  }
}

const NodeSwapQueue InplacePass::TryInplaceModifyVar(
    const std::string& var, const std::string& cache_var, const size_t& idx,
    ir::Graph* graph) const {
  PADDLE_ENFORCE(var_nodes_[var].size() >= 1 &&
                 var_nodes_[var].at(0)->Var() != nullptr);
  std::unique_ptr<VarDesc> var_desc(new VarDesc(*var_nodes_[var].at(0)->Var()));
  var_desc->SetName(cache_var);

  NodeSwapQueue swap_nodes;

  for (size_t i = idx; i < view_.AllOps().size(); ++i) {
    auto* op = view_.AllOps()[i];

    // redirect the input to the latest version of cache_var
    for (auto* node : op->inputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = graph->CreateVarNode(var_desc.get());

        // swap node to cache_node
        cache_node->outputs.insert(cache_node->outputs.end(),
                                   node->outputs.begin(), node->outputs.end());
        PADDLE_ENFORCE(node->inputs.size() == 1 && node->inputs[0]->IsOp());
        auto* prev_op = node->inputs[0];
        std::replace(prev_op->outputs.begin(), prev_op->outputs.end(), node,
                     cache_node);
        cache_node->inputs.emplace_back(prev_op);
        for (auto* next_op : node->outputs) {
          std::replace(next_op->inputs.begin(), next_op->inputs.end(), node,
                       cache_node);
        }

        swap_nodes.emplace_back(std::make_pair(node, cache_node));
      }
    }

    // if we need to rename the output,
    // always create a newer version of cache_var
    for (auto* node : op->outputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = graph->CreateVarNode(var_desc.get());
        // swap node to cache node
        cache_node->outputs.insert(cache_node->outputs.end(),
                                   node->outputs.begin(), node->outputs.end());
        cache_node->inputs.emplace_back(op);
        std::replace(op->outputs.begin(), op->outputs.end(), node, cache_node);
        for (auto* next_op : node->outputs) {
          std::replace(next_op->inputs.begin(), next_op->inputs.end(), node,
                       cache_node);
        }

        swap_nodes.emplace_back(std::make_pair(node, cache_node));
      }
    }
  }

  return swap_nodes;
}

void InplacePass::CommitModify(const NodeSwapQueue& swap_nodes,
                               ir::Graph* graph) const {
  for (auto& pair : swap_nodes) {
    auto *node = pair.first, *cache_node = pair.second;
    const std::string var = node->Name(), cache_var = cache_node->Name();
    var_nodes_[cache_var].emplace_back(cache_node);
    graph->RemoveNode(node);
    auto& nodes = var_nodes_.at(var);
    // release unused var in graph. Because python side memory optimize
    // may reused the var in same name, so we only clear the var node
    // after current inplaced index.
    nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
  }
}

void InplacePass::WithdrawModify(const NodeSwapQueue& nodes,
                                 ir::Graph* graph) const {
  for (auto& pair : nodes) {
    auto *node = pair.first, *cache_node = pair.second;
    const std::string var = node->Name(), cache_var = cache_node->Name();
    auto* prev_op = node->inputs[0];
    std::replace(prev_op->outputs.begin(), prev_op->outputs.end(), cache_node,
                 node);
    for (auto* next_op : node->outputs) {
      std::replace(next_op->inputs.begin(), next_op->inputs.end(), cache_node,
                   node);
    }
    graph->RemoveNode(cache_node);
  }
}

void InplacePass::TryInplaceOpInputOutput(ir::Node* op,
                                          ir::Graph* graph) const {
  VLOG(4) << "Try to inplace op " << op->Name();
  // some pre-requirments need to meet if the op want to inplaced.
  PADDLE_ENFORCE(op->Op() != nullptr, "op_desc is nullptr");

  auto* op_desc = op->Op();
  auto& infer_inplace =
      OpInfoMap::Instance().Get(op_desc->Type()).infer_inplace_;

  // 1. infer_inplace_ is registered.
  if (!static_cast<bool>(infer_inplace)) return;
  PADDLE_ENFORCE(static_cast<bool>(infer_inplace),
                 "%s's infer_inplace has not been registered", op_desc->Type());

  auto in_to_outs = infer_inplace(*op_desc);

  auto& all_ops = view_.AllOps();
  auto cursor = std::find(all_ops.begin(), all_ops.end(), op);
  size_t idx = std::distance(all_ops.begin(), cursor);

  for (auto& pair : in_to_outs) {
    auto& in_para_name = pair.first;
    auto& out_para_name = pair.second;

    auto input_vars = op->Op()->Input(in_para_name);
    if (!input_vars.size()) {
      VLOG(4) << "Parameter " << in_para_name << " is empty skip "
              << in_para_name << " => " << out_para_name << " pair";
      continue;
    }
    auto output_vars = op->Op()->Output(out_para_name);
    if (!output_vars.size()) {
      VLOG(4) << "Parameter " << out_para_name << " is empty skip "
              << in_para_name << " => " << out_para_name << " pair";
      continue;
    }
    auto in_var_name = input_vars.at(0);
    auto out_var_name = output_vars.at(0);
    auto* in_node = view_.GetNodeByName(in_var_name, op->inputs);
    auto* out_node = view_.GetNodeByName(out_var_name, op->outputs);

    VLOG(4) << "Try to inplace " << in_var_name << " with " << out_var_name;

    if (var_nodes_[in_var_name].back() != in_node) {
      VLOG(4) << "SKIP since " << in_var_name
              << " is also used as output by other ops";
      continue;
    }

    bool can_replace = true;
    if (in_var_name == out_var_name) {
      can_replace = false;
      VLOG(4) << "SKIP: Input variable " << in_var_name << " & Output variable "
              << out_var_name << " are the same";
    } else if (!NodeCanReused(in_node)) {
      can_replace = false;
      VLOG(4) << "SKIP: Input varialbe " << in_var_name << "cannot be reused";
    } else if (!NodeCanReused(out_node)) {
      can_replace = false;
      VLOG(4) << "SKIP: Output variable " << out_var_name
              << " cannot be reused";
    } else if (details::NodeSize(*in_node->Var()) !=
               details::NodeSize(*out_node->Var())) {
      can_replace = false;
      VLOG(4) << "SKIP: Input and Output varialbe size not match";
    }

    if (!can_replace) continue;

    // 2. there is no external pending op on the input node
    // if (view_.PendingOpsOnVar(in_node).size() > 1) {
    if (in_node->outputs.size() > 1 && !view_.CheckDeps(in_node, op)) {
      VLOG(4) << string::Sprintf(
          "Skiped pair %s => %s. %s input has external dependency."
          "inplace such pair will overwrite the memory.",
          out_var_name, in_var_name, op->Name());
      continue;
    }

    // 3. if output has been memory optimize by python(fluid.memory_optmize()).
    // this candidate  can not be inplaced. Will be deprecated in the future.
    if (view_.InSkipSet(out_node->Name())) {
      VLOG(4) << string::Sprintf(
          "Skiped %s => %s reused previous memory block in python memory "
          "optmize,"
          "it inplace may generate a circle",
          out_var_name, in_var_name, op->Name());
      continue;
    }

    // Debug Interface. Which would be skipped by the pass.
    if (out_node->Name() == FLAGS_memory_optimize_debug) {
      VLOG(3) << "Skiped var by force. FLAGS_memory_optimize_debug="
              << out_node->Name();
      continue;
    }

    // NOTE(dzhwinter):
    // two stage commit of inplaced process. if after inplace happens generate a
    // circle,
    // then withdraw the changes. Otherwise, safely add the node.
    auto swap_nodes =
        TryInplaceModifyVar(out_var_name, in_var_name, idx, graph);

    if (!ir::HasCircle(*graph)) {
      VLOG(3) << string::Sprintf("!!! %s,  %s => %s inplaced", op->Name(),
                                 out_var_name, in_var_name);
      InplaceModifyDesc(out_var_name, in_var_name, idx);
      CommitModify(swap_nodes, graph);
    } else {
      VLOG(3) << string::Sprintf(
          "Skiped pair %s => %s, inplace will generate a circle. withdraw %s",
          out_var_name, in_var_name, op->Name());
      WithdrawModify(swap_nodes, graph);
    }
  }
}

void GraphView::TopoSort(ir::Graph* graph) {
  //
  ops_.clear();
  auto deps_num = [](ir::Node* op) {
    auto cnt = 0;
    for (auto& var : op->inputs)
      if (var->inputs.size() > 0) ++cnt;
    return cnt;
  };

  std::queue<std::pair<ir::Node*, uint32_t>> ready_ops;

  int level = 0;
  auto nodes = graph->Nodes();
  std::unordered_map<ir::Node*, uint32_t> deps_map;
  for (auto& node : nodes) {
    if (node->IsOp() && node->Op() != nullptr) {
      deps_map[node] = deps_num(node);
      if (0 == deps_map[node]) {
        ready_ops.push({node, level});
      }
    }
  }

  while (!ready_ops.empty()) {
    auto item = ready_ops.front();
    ready_ops.pop();

    ops_.emplace_back(item.first);
    // record level when pop from queue
    op_level_[item.first] = item.second;

    for (auto node : item.first->outputs) {
      for (auto op : node->outputs) {
        --deps_map[op];
        if (deps_map[op] == 0) ready_ops.push({op, item.second + 1});
      }
    }
  }

  bool all_ops_checked = true;
  for (auto& node : nodes) {
    if (node->IsOp() && node->Op() != nullptr && deps_map[node] > 0) {
      all_ops_checked = false;
      break;
    }
  }

  PADDLE_ENFORCE(all_ops_checked, "All ops deps should be 0 after analysis");
}

// return true if current op node depeneds on all other op that use the same
// variable node
bool GraphView::CheckDeps(ir::Node* var, ir::Node* current_op) const {
  // get op list that rely on the same variable
  auto op_list = var->outputs;
  for (auto& op : op_list) {
    if (op == current_op) continue;

    VLOG(4) << "    GraphView::CheckDeps : " << op->Name() << "  & "
            << current_op->Name();
    if (!CheckOpDeps(op, current_op)) return false;
    VLOG(4) << "";
  }
  return true;
}

// check if op2 depends on op1's output
bool GraphView::CheckOpDeps(ir::Node* op1, ir::Node* op2) const {
  if (VLOG_IS_ON(4)) {
    auto print_op = [&](ir::Node* op, const char* name) {
      std::ostringstream os;
      os << "        " << name << " : " << op->Name() << " ";
      os << "Input args : ";
      for (auto& arg : op->inputs) os << arg->Name() << " ";
      os << "Output args : ";
      for (auto& arg : op->outputs) os << arg->Name() << " ";
      os << "Level : " << op_level_.at(op);
      VLOG(4) << os.str();
    };
    print_op(op1, "OP1");
    print_op(op2, "OP2");
  }
  if (op1 == op2) return true;
  if (op_level_.at(op1) >= op_level_.at(op2)) return false;

  for (auto& var : op2->inputs)
    if (var->inputs.size() > 0 && CheckOpDeps(op1, var->inputs[0])) return true;

  return false;
}

ir::Node* GraphView::GetNodeByName(const std::string& name,
                                   const std::vector<ir::Node*>& nodes) const {
  // nodes should be op->inputs/outputs
  // node in same node do have different name.
  std::unordered_set<std::string> nodes_in_op;
  bool has_dup_node =
      std::all_of(nodes.begin(), nodes.end(), [&nodes_in_op](ir::Node* node) {
        if (!node->IsVar() || node->IsCtrlVar() || node->Var() == nullptr) {
          if (nodes_in_op.count(node->Name())) return true;
          nodes_in_op.emplace(node->Name());
        }
        return false;
      });
  PADDLE_ENFORCE(has_dup_node == false, "nodes has same name!");
  ir::Node* node = nullptr;
  for (auto* it : nodes) {
    if (!it->IsVar() || it->IsCtrlVar() || it->Var() == nullptr) continue;
    if (it->Name() == name) {
      node = it;
      break;
    }
  }
  PADDLE_ENFORCE(node != nullptr,
                 string::Sprintf("Not found var %s in nodes!", name));
  return node;
}

std::vector<ir::Node*> GraphView::PendingOpsOnVar(ir::Node* node) {
  // get the pending ops depends on same var node.
  // because node also maybe a inplaced variable, so need to backtrack all the
  // previous inplaced vars.
  std::vector<ir::Node*> pending_ops;
  ir::Node* p = node;
  while (p != nullptr) {
    pending_ops.insert(pending_ops.end(), p->outputs.begin(), p->outputs.end());
    p = GetPrevCascadeInplacedVar(p);
  }
  return pending_ops;
}

void GraphView::Build(ir::Graph* g) {
  // track the var nodes in correct order.
  // Because we insert some new created node. Which may have data race between
  // nodes.
  // resolve data harzards depends on the var nodes in right order.
  TopoSort(g);

  // 2. track the nodes which used by parameter server.
  // these node can not be inplaced, otherwise trainer
  // pserver can not find each other name.
  auto update_skip_set = [&](ir::Node* node) {
    for (auto& in : node->inputs) {
      if (in->IsVar() && in->Var() != nullptr) dup_nodes_.emplace(in->Name());
    }
    for (auto& out : node->outputs) {
      if (out->IsVar() && out->Var() != nullptr)
        dup_nodes_.emplace(out->Name());
    }
  };
  for (auto& node : g->Nodes()) {
    if (!node->IsOp()) continue;
    if (node->Name() == "send") update_skip_set(node);
    if (node->Name() == "recv") update_skip_set(node);
    if (node->Name() == "prefetch") update_skip_set(node);
  }
}

const std::vector<ir::Node*>& GraphView::AllOps() { return ops_; }

bool GraphView::InSkipSet(const std::string& var) const {
  return dup_nodes_.count(var);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_pass, paddle::framework::details::InplacePass);
