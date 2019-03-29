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

#include "paddle/fluid/framework/details/memory_optimize_helper.h"
#include <algorithm>
#include <deque>
#include <functional>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/cpu_info.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/gpu_info.h"
#endif  // PADDLE_WITH_CUDA

namespace paddle {
namespace framework {
namespace details {
using paddle::framework::VarDesc;

std::vector<ir::Node*> SortOpLikeDescOrder(const ir::Graph& graph) {
  PADDLE_ENFORCE(graph.Has(kStaleProgramOpDescs),
                 "Graph has no attribute of kStaleProgramOpDescs.");
  // 1. get op desc order
  auto& op_descs = graph.Get<const std::vector<OpDesc*>>(kStaleProgramOpDescs);

  // 2. topology sort order
  auto nodes = graph.Nodes();
  std::deque<ir::Node*> ops;
  FilterVariables(nodes, [&](ir::Node* op) {
    if (op->IsOp() && op->Op() != nullptr) {
      ops.emplace_back(op);
    }
  });
  std::unordered_map<ir::Node*, size_t> op_deps;
  std::list<ir::Node*> ready_ops;
  std::unordered_map<ir::Node*, std::unordered_set<ir::Node*>> pending_ops;

  for (auto* op : ops) {
    std::unordered_set<ir::Node*> preceding_op;
    for (auto* in : op->inputs) {
      if (in->inputs.empty()) continue;
      PADDLE_ENFORCE(in->inputs.size() == 1 && in->inputs[0]->IsOp());
      preceding_op.emplace(in->inputs[0]);
      pending_ops[in->inputs[0]].emplace(op);
    }
    op_deps[op] = preceding_op.size();
    if (preceding_op.empty()) {
      ready_ops.emplace_back(op);
    }
  }

  // 3. generated op list based desc order and the topology order
  std::vector<ir::Node*> ret;
  std::list<OpDesc*> op_descs_list(op_descs.begin(), op_descs.end());

  auto update_by_found_node = [&](ir::Node* found_node) {
    for (auto* pending_op : pending_ops[found_node]) {
      if (--op_deps[pending_op] == 0) {
        ready_ops.emplace_back(pending_op);
      }
    }
    ready_ops.remove(found_node);
    ret.emplace_back(found_node);
  };

  while (!ready_ops.empty()) {
    bool all_of_ready_op_unmatched = true;
    for (auto it = op_descs_list.begin(); it != op_descs_list.end();) {
      auto op_desc = *it;
      ir::Node* found_node = nullptr;
      for (auto* op : ready_ops) {
        if (IsSameDesc(op->Op(), op_desc)) {
          found_node = op;
          break;
        }
      }

      // 3.1 op desc deleted by other pass
      if (found_node == nullptr) {
        ++it;
        continue;
      } else {
        all_of_ready_op_unmatched = false;
        it = op_descs_list.erase(it);
      }
      update_by_found_node(found_node);
    }

    // 3.2 op descs are added by other pass
    // preceding op non empty means some new op descs are
    // created, but not contained in return node list.
    // these new op desc may depend on each other.
    std::list<ir::Node*> prev_ready_ops(ready_ops);
    if (all_of_ready_op_unmatched) {
      for (auto op : prev_ready_ops) {
        update_by_found_node(op);
      }
    }
  }

  PADDLE_ENFORCE(std::all_of(
      op_deps.begin(), op_deps.end(),
      [&](const std::pair<ir::Node*, size_t>& p) { return p.second == 0; }));

  return ret;
}

size_t NodeSize(const VarDesc& node) {
  auto shape = node.GetShape();
  int size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  size_t type_size = SizeOfType(node.GetDataType());
  return type_size * std::abs(size);
}

size_t NodeSize(ir::Node* n) {
  VarDesc* desc = nullptr;
  // some op do not have block pointer
  if (n->inputs[0]->Op() != nullptr) {
    desc = FindVarDescInBlock(n);
  } else {
    desc = n->Var();
  }
  return NodeSize(*desc);
}

std::string DebugStringImpl(VarDesc* var) {
  std::stringstream ss;
  ss << var->Name();
  ss << "[";
  try {
    auto shape = var->GetShape();
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != shape.size() - 1) {
        ss << shape[i] << ",";
      } else {
        ss << shape[i];
      }
    }
    ss << "]";
  } catch (...) {
    ss << "Var has no VarDesc !!! Name:" << var->Name();
  }
  return ss.str();
}

std::string DebugString(ir::Node* var) {
  return DebugStringImpl(FindVarDescInBlock(var));
}

// NOTE(dzh): based ir node, if a large node has been reused
// by a small size node, then next time it appear in pool, it will
// have the small size. Find the original node shap from blockdesc.
VarDesc* FindVarDescInBlock(ir::Node* n) {
  PADDLE_ENFORCE(n->IsVar() && !n->IsCtrlVar() && n->inputs.size() == 1);
  BlockDesc* block = n->inputs[0]->Op()->Block();
  PADDLE_ENFORCE(block->HasVar(n->Name()),
                 string::Sprintf("Block do not has var %s", n->Name()));
  return block->FindVar(n->Name());
}

struct NodeComparator {
  bool operator()(ir::Node* lhs, ir::Node* rhs) const {
    auto* lhs_desc = FindVarDescInBlock(lhs);
    auto* rhs_desc = FindVarDescInBlock(rhs);
    // match data type
    if (lhs_desc->GetDataType() != rhs_desc->GetDataType()) {
      return false;
    }
    // match shape
    auto lhs_shape = lhs_desc->GetShape();
    auto rhs_shape = rhs_desc->GetShape();
    if ((lhs_shape[0] == -1 && rhs_shape[0] == -1) ||
        (lhs_shape[0] != -1 && rhs_shape[0] != -1)) {
      return NodeSize(lhs) == NodeSize(rhs);
    } else {
      return false;
    }
  }
};

void OrderedSet::Insert(ir::Node* var) {
  PADDLE_ENFORCE(var->IsVar() && !var->IsCtrlVar());
  if (mark_table_.count(var->Name()) != 0) {
    mark_table_[var->Name()]->emplace_back(var);
    return;
  }

  auto* var_desc = FindVarDescInBlock(var);
  auto var_shape = var_desc->GetShape();
  int batch_size = static_cast<int>(var_shape[0]);

  NodeComparator functor;
  Iter it = nodes_.begin();
  while (it != nodes_.end()) {
    auto& prev = it->front();
    auto* cache_desc = FindVarDescInBlock(prev);
    int cache_batch_size = cache_desc->GetShape()[0];
    if ((cache_batch_size == -1 && batch_size == -1) ||
        (cache_batch_size != -1 && batch_size != -1)) {
      if (functor(prev, var)) {
        ++it;
      } else {
        break;
      }
    } else if (cache_batch_size == -1 && batch_size != -1) {
      ++it;
    } else if (cache_batch_size != -1 && batch_size == -1) {
      break;
    }
  }

  it = nodes_.insert(it, {var});
  mark_table_[var->Name()] = it;
}

int OrderedSet::GetNodeIndexInPool(ir::Node* var) {
  return std::distance(nodes_.begin(), mark_table_[var->Name()]);
}

ir::Node* OrderedSet::FindBestFitNode(ir::Node* var) const {
  ir::Node* found_node = nullptr;
  NodeComparator functor;

  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    auto& candidate = it->front();
    if (functor(var, candidate)) {
      found_node = candidate;
      break;
    }
  }
  return found_node;
}

ir::Node* OrderedSet::FindNextBestFitNode(ir::Node* var, ir::Node* prev) const {
  ir::Node* found_node = nullptr;
  NodeComparator functor;
  auto it =
      std::find_if(nodes_.begin(), nodes_.end(), [&](const NodeVector& v) {
        if (v.front() == prev)
          return true;
        else
          return false;
      });
  PADDLE_ENFORCE(it != nodes_.end(), "Not found previous in node list!");
  for (it = std::next(it); it != nodes_.end(); ++it) {
    auto& candidate = it->front();
    if (functor(var, candidate)) {
      found_node = candidate;
      break;
    }
  }
  return found_node;
}

bool OrderedSet::Has(ir::Node* var) const {
  if (mark_table_.count(var->Name())) {
    auto& node_in_samename = mark_table_.at(var->Name());
    auto iter =
        std::find_if(node_in_samename->begin(), node_in_samename->end(),
                     [&](ir::Node* n) { return n->Name() == var->Name(); });
    return iter != node_in_samename->end();
  }
  return false;
}

void OrderedSet::Erase(const std::string& var) {
  PADDLE_ENFORCE(mark_table_.count(var));
  nodes_.erase(mark_table_[var]);
  mark_table_.erase(var);
}

void OrderedSet::Erase(ir::Node* var) {
  PADDLE_ENFORCE(var != nullptr);
  Erase(var->Name());
}

std::string OrderedSet::ToString() const {
  std::stringstream ss;
  for (auto it = nodes_.begin(); it != nodes_.end(); ++it) {
    for (auto& node : *it) {
      ss << DebugString(node) << " ";
    }
  }
  return ss.str();
}

bool NodeCanReused(ir::Node* node) {
  // valid the node is a var node
  // vars can be @EMPTY@, @LR_DECAY_REUSE_ID@. For example, while_grad
  if (node == nullptr || !node->IsVar() || node->IsCtrlVar() ||
      node->Name() == kEmptyVarName)
    return false;

  bool flag = true;
  // op output force generated in cpu, can not be reused.
  for (auto* op : node->inputs) {
    if (op->Op()->HasAttr("force_cpu")) {
      flag &= framework::AttrReader(op->Op()->GetAttrMap())
                  .Get<bool>("force_cpu") == 0;
    }
  }
  // var desc validation.
  flag &= NodeCanReused(*node->Var());
  return flag;
}

int MinChunkSize() {
  int size{0};
#ifdef PADDLE_WITH_CUDA
  size = platform::GpuMinChunkSize();
#else
  size = platform::CpuMinChunkSize();
#endif  // PADDLE_WITH_CUDA
  return size;
}

bool NodeCanReused(const VarDesc& node) {
  auto type = node.GetType();
  // only these types holds bulk of gpu memory
  if (!(type == proto::VarType::LOD_TENSOR ||
        type == proto::VarType::LOD_TENSOR_ARRAY)) {
    return false;
  }
  // persistable variable is parameter
  if (node.Persistable()) {
    return false;
  }
  // shape < min_chunk_size is meaningless.
  // further more, fetched loss always has size = 1
  // which should not be reused.
  auto shape = node.GetShape();
  int size = std::abs(
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  if (shape.empty() || size < MinChunkSize()) {
    return false;
  }
  return true;
}

bool OpHasSubBlock(OpDesc* desc) {
  const AttributeMap& attrs = desc->GetAttrMap();
  for (auto& attr : attrs) {
    if (attr.second.type() == typeid(BlockDesc*) ||             // NOLINT
        attr.second.type() == typeid(std::vector<BlockDesc*>))  // NOLINT
      return true;
  }
  return false;
}

ControlFlowGraph::ControlFlowGraph(const ir::Graph& graph) {
  ops_ = SortOpLikeDescOrder(graph);
  ConnectNodes();
}

void ControlFlowGraph::BuildCFGGraph() {
  // FIXME(dzh): same effect with ConnectNodes, but use the control
  // link to build dependency graph, it goes wrong in transformer.
  for (ir::Node* op : ops_) {
    for (auto& input_var : op->inputs) {
      if (!input_var->inputs.empty()) {
        PADDLE_ENFORCE(
            input_var->inputs.size() == 1 && input_var->inputs[0]->IsOp(),
            "Preceding Op Node of Var Node must be unique");
        auto* pred_op = input_var->inputs[0];
        if (pred_op->Op() != nullptr) {
          predecessors_[op].insert(pred_op);
          successors_[pred_op].insert(op);
        }
      }
      if (input_var->IsVar() && !input_var->IsCtrlVar()) {
        uses_[op].insert(input_var->Name());
      }
    }
    for (auto& output_var : op->outputs) {
      // output var may be used by many op
      for (auto* succ_op : output_var->outputs) {
        if (succ_op->Op() != nullptr) {
          successors_[op].insert(succ_op);
          predecessors_[succ_op].insert(op);
        }
      }
      if (output_var->IsVar() && !output_var->IsCtrlVar()) {
        defs_[op].insert(output_var->Name());
      }
    }
  }
}

void ControlFlowGraph::ConnectNodes() {
  for (size_t i = 0; i < ops_.size(); ++i) {
    auto& op = ops_[i];
    try {
      auto& next_op = ops_.at(i + 1);
      successors_[op].insert(next_op);
      predecessors_[next_op].insert(op);
    } catch (...) {
      // do nothing
    }

    FilterVariables(op->inputs,
                    [&](ir::Node* var) { uses_[op].emplace(var->Name()); });

    FilterVariables(op->outputs,
                    [&](ir::Node* var) { defs_[op].emplace(var->Name()); });
  }
}

void ControlFlowGraph::LiveVariableAnalysis() {
  // NOTE(dzh): variable liveless analysis (a.k.a reversed_ops algorithm)
  // compute the liveness of for each variable though reversed_ops algorithm.
  // It iterates the operators from end to begin, compute the live in/live out
  // variable set for each op, then the diff between in/out will be used for
  // the variable reuse. For detail refer to
  // http://www.cs.cornell.edu/courses/cs4120/2013fa/lectures/lec26-fa13.pdf
  std::list<ir::Node*> work_list(ops_.rbegin(), ops_.rend());
  while (!work_list.empty()) {
    ir::Node* op = work_list.front();
    work_list.pop_front();
    // get the live_in calculated before. Empty if first.
    auto prev_live_in = std::move(live_in_[op]);
    for (auto& s : successors_[op]) {
      for (auto& var : live_in_[s]) {
        live_out_[op].insert(var);
      }
    }
    for (auto& var : uses_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : live_out_[op]) {
      live_in_[op].insert(var);
    }
    for (auto& var : defs_[op]) {
      if (uses_[op].count(var)) continue;
      live_in_[op].erase(var);
    }

    // If the live_in is not changed, then the liveness analysis of
    // predecessors is completed.
    //
    // Otherwise, recalculate the predecessors liveness
    if (live_in_[op] != prev_live_in) {
      for (auto& pre : predecessors_[op]) {
        work_list.push_back(pre);
      }
    }
  }

  for (auto* op : ops_) {
    unlived_vars_[op] = std::set<std::string>();
    for (auto& var : this->LiveIn(op)) {
      if (!this->LiveOut(op).count(var)) {
        unlived_vars_[op].insert(var);
      }
    }
  }
}

void ControlFlowGraph::RenameVarInCFGGraph(const std::string& old_node,
                                           const std::string& new_node,
                                           int begin_idx) {
  std::vector<bool> need_update(ops_.size(), false);
  // update graph from begin idx to the end
  for (size_t i = begin_idx; i != ops_.size(); ++i) {
    auto* op = ops_[i];
    if (uses_[op].find(old_node) != uses_[op].end()) {
      uses_[op].erase(old_node);
      uses_[op].insert(new_node);
    }
    if (defs_[op].find(old_node) != defs_[op].end()) {
      defs_[op].erase(old_node);
      defs_[op].insert(new_node);
    }
    if (live_in_[op].find(old_node) != live_in_[op].end()) {
      live_in_[op].erase(old_node);
      live_in_[op].insert(new_node);
      need_update[i] = true;
    }
    if (live_out_[op].find(old_node) != live_out_[op].end()) {
      live_out_[op].erase(old_node);
      live_out_[op].insert(new_node);
      need_update[i] = true;
    }
  }

  for (size_t i = begin_idx; i < ops_.size(); ++i) {
    if (!need_update[i]) continue;
    auto* op = ops_[i];
    for (auto& var : this->LiveIn(op)) {
      if (!this->LiveOut(op).count(var)) {
        unlived_vars_[op].insert(var);
      }
    }
  }
}

const std::set<std::string>& ControlFlowGraph::LiveIn(ir::Node* op) const {
  auto it = live_in_.find(op);
  PADDLE_ENFORCE(
      it != live_in_.end(),
      string::Sprintf("Expect %s in live_in, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string>& ControlFlowGraph::LiveOut(ir::Node* op) const {
  auto it = live_out_.find(op);
  PADDLE_ENFORCE(
      it != live_out_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string>& ControlFlowGraph::Use(ir::Node* op) const {
  auto it = uses_.find(op);
  PADDLE_ENFORCE(
      it != uses_.end(),
      string::Sprintf("Expect %s in use, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string>& ControlFlowGraph::Unlived(ir::Node* op) const {
  auto it = unlived_vars_.find(op);
  PADDLE_ENFORCE(
      it != unlived_vars_.end(),
      string::Sprintf("Expect %s in unlived_set, but Not Found.", op->Name()));
  return it->second;
  return it->second;
}

const std::vector<ir::Node*>& ControlFlowGraph::Ops() const { return ops_; }

std::vector<ir::Node*>& ControlFlowGraph::Ops() { return ops_; }

ir::Node* ControlFlowGraph::GetNodeByName(const std::string& name,
                                          ir::Node* op) const {
  // in ssa-graph, different version nodes have same name,
  // this function get the latest version var before target op
  // It may return nullptr, such as data node.
  ir::Node* found_node = nullptr;
  for (auto* node : ops_) {
    if (node == op) break;
    for (auto& output : node->outputs) {
      PADDLE_ENFORCE((output != nullptr && output->IsVar()),
                     "Output is empty!");
      if (output->Var() && output->Name() == name) {
        found_node = output;
      }
    }
  }
  return found_node;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
