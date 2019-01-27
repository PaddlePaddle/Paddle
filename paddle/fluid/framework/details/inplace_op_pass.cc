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
#include <stack>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/details/graph_print_pass.h"
#include "paddle/fluid/framework/details/memory_optimize_pass.h"
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

// For backward compacity. if enable_inplace_whitelist is turn on.
// only the ops in whitelist will be use inplace strategy.
// if not, all the op will be inplaced if it registered with InplaceClass
DEFINE_bool(
    enable_inplace_whitelist, true,
    "If this option turns on, only these op in whitelist can be inplaced."
    "If it turns off, all of the running op can be candidate of inplaced op."
    "Such as scale, elementwise_add"
    "By default, it's turned on");

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

static inline std::string NodeDebugString(ir::Node* var) {
  std::ostringstream os;
  if (var->IsCtrlVar()) {
    os << "kControlDepVarName"
       << " ";
  } else if (var->IsOp()) {
    os << "kOperation"
       << " " << var->Name();
    PADDLE_ENFORCE(var->Op() != nullptr && var->Op()->Type() == var->Name());
  } else if (var->IsVar()) {
    os << "kVariable"
       << " " << var->Name();
    PADDLE_ENFORCE(var->Var() != nullptr && var->Var()->Name() == var->Name());
  } else {
    PADDLE_THROW("Unknown node type.");
  }
  return os.str();
}

static inline std::string OpDebugString(ir::Node* var) {
  ir::Node* op = var;
  if (var->IsVar()) op = var->inputs.at(0);
  std::stringstream os;
  os << op->Name() << " : ";

  os << "Input ";
  VLOG(3) << op->Name();
  for (auto* var : op->inputs) {
    if (var->IsVar() && !var->IsCtrlVar()) {
      PADDLE_ENFORCE(var->Var() != nullptr && var->Var()->Name() == var->Name(),
                     "unmatched desc and var");
      // os << var << ":" << var->Name() << " ";
      os << var->Name() << " ";
    }
  }
  os << "Output ";
  VLOG(3) << op->Name();
  for (auto* var : op->outputs) {
    VLOG(3) << var;
    VLOG(3) << var->Name();
    if (!var->IsVar()) {
      VLOG(3) << "error";
    }
    // VLOG(3) << var->Var()->Name();
    if (var->IsVar() && !var->IsCtrlVar()) {
      PADDLE_ENFORCE(var->Var() != nullptr && var->Var()->Name() == var->Name(),
                     "unmatched desc and var");
      // os << var << ":" << var->Name() << " ";
      os << var->Name() << " ";
    }
    if (var->Name() == "fc_10.tmp_0") {
      VLOG(3) << NodeDebugString(var);
    }
  }
  return os.str();
}

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

template <typename Container>
static inline bool ConnectByCtrlVar(const Container& group1,
                                    const Container& group2) {
  bool connected = false;
  std::unordered_set<ir::Node*> outputs;
  for (auto* op : group1) {
    for (auto* var : op->outputs) {
      if (var->IsCtrlVar()) outputs.emplace(var);
    }
  }
  for (auto* op : group2) {
    for (auto* var : op->inputs) {
      if (outputs.count(var)) connected = true;
    }
  }
  return connected;
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

std::unique_ptr<ir::Graph> InplacePass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  var_nodes_.clear();
  view_.Build(graph.get());
  InitSSAGraphNodes();

  std::unique_ptr<SSAGraphPrinter> printer(new SSAGraphPrinterImpl);

  for (auto* op : view_.AllOps()) {
    if (FLAGS_enable_inplace_whitelist && !whitelist_.count(op->Name()))
      continue;
    TryInplaceOpInputOutput(op, graph.get());
  }
  graph->ResolveHazard(var_nodes_);

  constexpr char graph_path[] = "ir_graph_inplaced.txt";
  std::unique_ptr<std::ostream> fout(new std::ofstream(graph_path));
  PADDLE_ENFORCE(fout->good());
  printer->Print(*graph, *fout);
  // for(auto* op : view_.AllOps()) {
  //   VLOG(3) << OpDebugString(op);
  // }
  return graph;
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
    if (op_desc->Block()->HasVar(var)) op_desc->Block()->RemoveVar(var);
    op_desc->Flush();
  }
}

void InplacePass::InplaceModifyVar(const std::string& var,
                                   const std::string& cache_var,
                                   const size_t& idx, ir::Graph* graph) const {
  PADDLE_ENFORCE(var_nodes_[var].size() >= 1 &&
                 var_nodes_[var].at(0)->Var() != nullptr);
  std::unique_ptr<VarDesc> var_desc(new VarDesc(*var_nodes_[var].at(0)->Var()));
  var_desc->SetName(cache_var);

  for (size_t i = idx; i < view_.AllOps().size(); ++i) {
    auto* op = view_.AllOps()[i];

    // redirect the input to the latest version of cache_var
    for (auto* node : op->inputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = graph->CreateVarNode(var_desc.get());
        var_nodes_[cache_var].emplace_back(cache_node);

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

        // release unused var in graph. Because python side memory optimize
        // may reused the var in same name, so we only clear the var node
        // after current inplaced index.
        graph->RemoveNode(node);
        auto& nodes = var_nodes_.at(var);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
      }
    }

    // if we need to rename the output,
    // always create a newer version of cache_var
    for (auto* node : op->outputs) {
      if (node->Name() == var) {
        ir::Node* cache_node = graph->CreateVarNode(var_desc.get());
        var_nodes_[cache_var].emplace_back(cache_node);
        // swap node to cache node
        cache_node->outputs.insert(cache_node->outputs.end(),
                                   node->outputs.begin(), node->outputs.end());
        cache_node->inputs.emplace_back(op);
        std::replace(op->outputs.begin(), op->outputs.end(), node, cache_node);
        for (auto* next_op : node->outputs) {
          std::replace(next_op->inputs.begin(), next_op->inputs.end(), node,
                       cache_node);
        }

        // release unsed var in graph
        graph->RemoveNode(node);
        auto& nodes = var_nodes_.at(var);
        nodes.erase(std::remove(nodes.begin(), nodes.end(), node), nodes.end());
      }
    }
  }
}

void InplacePass::TryInplaceOpInputOutput(ir::Node* op,
                                          ir::Graph* graph) const {
  PADDLE_ENFORCE(op->Op() != nullptr && op->Op()->Block() != nullptr,
                 "op_desc is nullptr");
  // 3 pre-requirments need to meet if the op want to inplaced.
  // 1. infer_inplace_ is registered.
  auto* op_desc = op->Op();
  auto& infer_inplace =
      OpInfoMap::Instance().Get(op_desc->Type()).infer_inplace_;
  if (!static_cast<bool>(infer_inplace)) return;
  PADDLE_ENFORCE(static_cast<bool>(infer_inplace),
                 "%s's infer_inplace has not been registered", op_desc->Type());

  auto* block = op_desc->Block();
  auto in_to_outs = infer_inplace(*op_desc, block);

  auto& all_ops = view_.AllOps();
  auto cursor = std::find(all_ops.begin(), all_ops.end(), op);
  size_t idx = std::distance(all_ops.begin(), cursor);
  VLOG(3) << op->Name() << idx;

  for (auto& pair : in_to_outs) {
    auto& in_var_name = pair.first;
    auto& out_var_name = pair.second;
    auto* in_node = view_.GetNodeByName(in_var_name, op->inputs);
    auto* out_node = view_.GetNodeByName(out_var_name, op->outputs);
    // 2. there is no external pending op on the input node
    if (view_.PendingOpsOnVar(in_node).size() > 1) {
      VLOG(3) << string::Sprintf(
          "!!! %s input has external dependency, can not inplaced,  %s => %s "
          "skiped",
          op->Name(), out_var_name, in_var_name);
      continue;
    }
    // 3. if output reuse input inplaced, the dependency group is not changed.
    // For detail, check
    // the function description in "OutConnectInputByCtrlVar"
    if (view_.OutConnectInputByCtrlVar(in_node, out_node)) {
      VLOG(3) << string::Sprintf(
          "!!! %s input output connect by ctrl var, cannot inplaced,  %s => %s "
          "skiped",
          op->Name(), out_var_name, in_var_name);
      continue;
    }
    VLOG(3) << string::Sprintf("!!! %s,  %s => %s inplaced", op->Name(),
                               out_var_name, in_var_name);
    // VLOG(3) << "Out " << OpDebugString(op);
    InplaceModifyDesc(out_var_name, in_var_name, idx);
    InplaceModifyVar(out_var_name, in_var_name, idx, graph);
  }
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

void GraphView::Build(ir::Graph* g) { ops_ = SortOpLikeDescOrder(*g); }

const std::vector<ir::Node*> GraphView::AllOps() { return ops_; }

bool GraphView::OutConnectInputByCtrlVar(ir::Node* in_var, ir::Node* out_var) {
  // assume v_a0, v_a1 is variable. v_a0 -> v_a0 means already inplaced.
  // v_a1 -> v_a1 means already inplaced.
  // Currently we make decision to check if the v_a0 -> v_a1 can be inplace.
  //
  // v_a0
  //  +
  //  |
  //  v
  // v_a0
  //  +
  //  |
  //  v
  // v_a1
  //  +
  //  |
  //  v
  // v_a1
  // start from the first inplaced input v_a0(on the top one).
  // Do a DFSSearch, get all its paths. If there is one path connect
  // the in_var and out_var which contains control dep var.
  // Means there a control path. out_var can not be inplaced use in_var.

  std::unordered_set<ir::Node *> out_var_set, in_var_set;
  ir::Node* out = out_var;
  // get the ops with same output name
  while (out != nullptr) {
    out_var_set.emplace(out);
    out = GetNextCascadeInplacedVar(out);
  }

  // get ops with same input name
  ir::Node* in = in_var;
  while (in != nullptr) {
    in_var_set.emplace(in);
    in = GetPrevCascadeInplacedVar(in);
  }
  // find if there is path with control dep var connect the in_var_set and
  // out_var_set
  return ConnectByCtrlVar(in_var_set, out_var_set);
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(inplace_pass, paddle::framework::details::InplacePass);
