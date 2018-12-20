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

#include "paddle/fluid/framework/details/analysis_var_pass.h"
#include <algorithm>
#include <atomic>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

DEFINE_bool(enable_subgraph_optimize, false,
            "SubGraph also reuse global graph variables, it will reduce the "
            "memory occupation"
            "but a higher risk of memory reuse error. default disabled.");
DEFINE_string(memory_optimize_debug, "",
              "debug the operator output variable when do the variable reuse."
              "memory reuse pass."
              "only for debug, default disabled.");

namespace paddle {
namespace framework {
namespace details {

static inline bool IsSameDesc(OpDesc* op1, OpDesc* op2) {
  return op1->Type() == op2->Type() && op1->Inputs() == op2->Inputs() &&
         op1->Outputs() == op2->Outputs();
}

template <typename Container, typename Callback>
class FilterVariableImpl {
 public:
  void operator()(const Container& nodes, Callback callback) {
    for (auto* node : nodes) {
      callback(node);
    }
  }
};

// filter var node for op->inputs/outputs
template <typename Callback>
class FilterVariableImpl<std::vector<ir::Node*>, Callback> {
 public:
  void operator()(const std::vector<ir::Node*>& nodes, Callback callback) {
    for (auto* var : nodes) {
      if (var->IsVar() && !var->IsCtrlVar()) {
        callback(var);
      }
    }
  }
};

template <typename Container, typename Callback>
void FilterVariables(const Container& nodes, Callback callback) {
  FilterVariableImpl<Container, Callback>()(nodes, callback);
}

std::unique_ptr<ir::Graph> AnalysisVarPass::ApplyImpl(
    std::unique_ptr<ir::Graph> graph) const {
  auto nodes = graph->Nodes();
  auto subblock_vars = GetSubBlockVars(nodes);
  skip_set_.insert(subblock_vars.begin(), subblock_vars.end());

  cfg_.reset(new details::ControlFlowGraph(*graph));
  cfg_->LiveVariableAnalysis();
  InitSSAGraphNodes();

  int reuse_id = 0;
  for (size_t idx = 0; idx < cfg_->Ops().size(); ++idx) {
    auto& op = cfg_->Ops()[idx];
    auto* op_desc = op->Op();
    // some op in graph has no op desc
    if (op_desc == nullptr) continue;
    if (OpHasSubBlock(op_desc)) {
      if (FLAGS_enable_subgraph_optimize) {
        SubGraphOptimize(op_desc);
      } else {
        VLOG(3) << op->Name()
                << " has subblock, but disable subgraph optimize. skipped.";
        continue;
      }
    }

    for (auto& var : op->outputs) {
      if (NodeCanReused(var) && cfg_->Use(op).count(var->Name()) == 0) {
        ir::Node* cache = pool_.NodeMatch(var);
        if (var->Name() == FLAGS_memory_optimize_debug) {
          VLOG(3) << "start match var " << DebugString(var) << " of op "
                  << op->Name();
          VLOG(3) << pool_.ToString();
          VLOG(3) << "matched in pool : "
                  << ((cache == nullptr) ? "False" : "True");
        }
        if (cache != nullptr) {
          if (var->Name() == cache->Name()) {
            VLOG(3) << "The same cache variable is cascade reused."
                    << var->Name() << " is re-filled to the pool after"
                    << "the reused op is finished. Current op can not "
                    << "replace it again. Skip this candidate.";
            continue;
          }

          int node_idx_in_pool = pool_.GetIndex(cache);
          VLOG(3) << string::Sprintf(
              "!!! %s,  %s => %s, cache idx %d, pool size %d",
              std::to_string(reuse_id++), DebugString(var), DebugString(cache),
              node_idx_in_pool, static_cast<int>(pool_.size()));
          // update CFG Graph on the fly.
          // reused var maybe re-fill into the pool
          cfg_->RenameVarInCFGGraph(var->Name(), cache->Name(), idx);
          // NOTE(dzhwinter): we need to both update the ProgramDesc
          // and IR Graph. because op_desc/var_desc is used in CreateOp,
          // CreateVar when running happens. But IR Graph
          // define the dependence relationship between nodes.
          RenameVarInGraphDesc(var->Name(), cache->Name(), idx);
          RenameVarInGraphNode(var->Name(), cache->Name(), idx, graph.get());

          pool_.Erase(cache);
        }
      }
    }
    // fill the pool
    for (auto var : cfg_->LiveIn(op)) {
      if (cfg_->LiveOut(op).count(var) == 0) {
        ir::Node* var_node = cfg_->GetNodeFromVarName(var, op);
        if (var_node == nullptr) continue;
        if (NodeCanReused(var_node) && !pool_.Has(var_node)) {
          pool_.Insert(var_node, op);
        }
      }
    }
  }
  graph->ResolveHazard(var_nodes_);

  // For early delete pass. use GraphNodePool load the unlived vars.
  // 1. find all deps op for each unlived var in memory pool.
  for (auto& op : graph->Nodes()) {
    for (auto& var : op->inputs) {
      if (pool_.Has(var)) {
        pool_.Insert(var, op);
      }
    }
  }
  // 2. convert ir node based memory pool to graph node
  // because Node* maybe released bettwen passes.
  auto& graph_pool = graph->Get<GraphNodePool>(kGraphNodePool);
  for (auto it = pool_.begin(); it != pool_.end(); ++it) {
    std::unordered_set<OpDesc*> descs;
    for (auto& op : it->second) {
      PADDLE_ENFORCE(op->IsOp());
      descs.insert(op->Op());
    }
    graph_pool.push_back(std::make_pair(it->first->Name(), descs));
  }

  return graph;
}

void AnalysisVarPass::SubGraphOptimize(OpDesc* op_desc) const {
  // conditional block, while op and their grad op
  auto* sub_block_desc =
      AttrReader(op_desc->GetAttrMap()).Get<BlockDesc*>("sub_block");

  // create a mirror block to construct an IR Graph.
  ProgramDesc prog;
  auto* copy_block = prog.MutableBlock(0);
  for (auto* op : sub_block_desc->AllOps()) {
    auto* copy_op = copy_block->AppendOp();
    copy_op->CopyFrom(*op);
    copy_op->Flush();
  }

  for (auto* var : sub_block_desc->AllVars()) {
    auto* copy_var = copy_block->Var(var->Name());
    copy_var->SetDataType(var->GetDataType());
    // only lod tensor can be reused. So ignore the multiple dims case.
    copy_var->SetType(var->GetType());
    copy_var->SetShape(var->GetShape());
    copy_var->SetPersistable(var->Persistable());
  }

  ir::Graph sub_graph(prog);
  std::unordered_set<ir::Node*> sub_graph_all_ops;
  FilterVariables(sub_graph.Nodes(), [&](ir::Node* var) {
    // sub_graph_all_ops.emplace(var);
    if (var->IsVar() && !var->IsCtrlVar()) {
      sub_graph_all_ops.emplace(var);
    }
  });
  int sub_reuse_id = 0;
  // subgraph nodes is unordered, reuse need to follow the desc order.
  // find the right op node through the descs
  for (auto* sub_op_desc : sub_block_desc->AllOps()) {
    ir::Node* sub_op = nullptr;
    for (auto* node : sub_graph_all_ops) {
      if (node->Op() == sub_op_desc) {
        sub_op = node;
        break;
      }
    }
    PADDLE_ENFORCE(sub_op != nullptr);
    for (auto* var : sub_op->outputs) {
      if (NodeCanReused(var)) {
        ir::Node* cache = pool_.NodeMatch(var);
        if (cache != nullptr) {
          if (var->Var()->GetDataType() != cache->Var()->GetDataType()) {
            continue;
          }
          int node_idx_in_pool = pool_.GetIndex(cache);
          VLOG(3) << string::Sprintf(
              "!!! %s,  %s => %s, cache idx %d, pool size %d",
              std::to_string(sub_reuse_id++), DebugString(var),
              DebugString(cache), node_idx_in_pool,
              static_cast<int>(pool_.size()));
          // NOTE(dzh): subblock is not in IR graph. Modify the block_desc
          // immediately to make the subblock variable reuse strategy take
          // effect. Because it is a single op in graph. No need to
          // update the ir nodes.
          sub_op_desc->Rename(var->Name(), cache->Name());
          if (sub_op_desc->Block()->HasVar(var->Name())) {
            sub_op_desc->Block()->RemoveVar(var->Name());
          }
        }
      }
    }
  }
}

std::unordered_set<std::string> AnalysisVarPass::GetSubBlockVars(
    const std::unordered_set<ir::Node*>& nodes) const {
  std::unordered_set<std::string> vars;
  for (auto& op : nodes) {
    if (!op->IsOp() || op->Op() == nullptr) continue;
    auto* op_desc = op->Op();
    if (OpHasSubBlock(op_desc)) {
      auto inputs = op_desc->InputArgumentNames();
      auto outputs = op_desc->OutputArgumentNames();
      vars.insert(inputs.begin(), inputs.end());
      vars.insert(outputs.begin(), outputs.end());
    }
  }
  return vars;
}

void AnalysisVarPass::RenameVarInGraphDesc(const std::string& var,
                                           const std::string& cache_var,
                                           size_t idx) const {
  for (size_t i = idx; i < cfg_->Ops().size(); ++i) {
    auto* op = cfg_->Ops()[i];
    PADDLE_ENFORCE(op->IsOp() && op->Op());
    auto* op_desc = op->Op();
    op_desc->RenameInput(var, cache_var);
    op_desc->RenameOutput(var, cache_var);
    if (op_desc->Block()->HasVar(var)) op_desc->Block()->RemoveVar(var);
    op_desc->Flush();
  }
}

void AnalysisVarPass::InitSSAGraphNodes() const {
  std::unordered_map<std::string, std::unordered_set<ir::Node*>> all_vars;
  if (var_nodes_.empty()) {
    for (auto* op : cfg_->Ops()) {
      for (auto* node : op->inputs) {
        if (all_vars[node->Name()].count(node) == 0) {
          all_vars[node->Name()].emplace(node);
          var_nodes_[node->Name()].emplace_back(node);
        }
      }
      for (auto* node : op->outputs) {
        if (all_vars[node->Name()].count(node) == 0) {
          all_vars[node->Name()].emplace(node);
          var_nodes_[node->Name()].emplace_back(node);
        }
      }
    }
  }
}

void AnalysisVarPass::RenameVarInGraphNode(const std::string& var,
                                           const std::string& cache_var,
                                           size_t idx, ir::Graph* graph) const {
  // if replace happens, we need to create a newer version cache_var
  // but use the same dims/data_type with var.
  PADDLE_ENFORCE(var_nodes_[var].size() >= 1 &&
                 var_nodes_[var].at(0)->Var() != nullptr);
  std::unique_ptr<VarDesc> var_desc(new VarDesc(*var_nodes_[var].at(0)->Var()));
  var_desc->SetName(cache_var);

  for (size_t i = idx; i < cfg_->Ops().size(); ++i) {
    auto* op = cfg_->Ops()[i];

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
      }
    }
  }

  // release node of unused var in graph
  for (auto* node : var_nodes_[var]) {
    graph->RemoveNode(node);
  }
  var_nodes_.at(var).clear();
}

bool AnalysisVarPass::NodeCanReused(ir::Node* node) const {
  if (!node->IsVar() || node->IsCtrlVar()) return false;
  auto* desc = node->Var();
  auto type = desc->GetType();
  if (desc->Persistable() || type != proto::VarType::LOD_TENSOR ||
      desc->GetShape().empty()) {
    return false;
  }
  // vars can be @EMPTY@, @LR_DECAY_REUSE_ID@. For example, while_grad
  std::string name = node->Name();
  if (!name.empty() && name[0] == '@' && name[name.size() - 1] == '@')
    return false;
  if (skip_set_.count(name)) return false;
  for (auto* op : node->inputs) {
    if (op->Op()->HasAttr("force_cpu")) {
      // op output force generated in cpu, can not be reused.
      return framework::AttrReader(op->Op()->GetAttrMap())
                 .Get<bool>("force_cpu") == 0;
    }
  }
  return true;
}

bool AnalysisVarPass::OpHasSubBlock(OpDesc* desc) const {
  const AttributeMap& attrs = desc->GetAttrMap();
  for (auto& attr : attrs) {
    if (attr.second.type() == typeid(BlockDesc*) ||             // NOLINT
        attr.second.type() == typeid(std::vector<BlockDesc*>))  // NOLINT
      return true;
  }
  return false;
}

std::vector<ir::Node*> SortOpLikeDescOrder(const ir::Graph& graph) {
  PADDLE_ENFORCE(graph.Has(kAllOpDescs),
                 "Graph has no attribute of kAllOpDescs.");
  // 1. get op desc order
  auto& op_descs = graph.Get<const std::vector<OpDesc*>>(kAllOpDescs);

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
}

void ControlFlowGraph::RenameVarInCFGGraph(const std::string& old_node,
                                           const std::string& new_node,
                                           int begin_idx) {
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
    }
    if (live_out_[op].find(old_node) != live_out_[op].end()) {
      live_out_[op].erase(old_node);
      live_out_[op].insert(new_node);
    }
  }
}

const std::set<std::string> ControlFlowGraph::LiveIn(ir::Node* op) const {
  auto it = live_in_.find(op);
  PADDLE_ENFORCE(
      it != live_in_.end(),
      string::Sprintf("Expect %s in live_in, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string> ControlFlowGraph::LiveOut(ir::Node* op) const {
  auto it = live_out_.find(op);
  PADDLE_ENFORCE(
      it != live_out_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}

const std::set<std::string> ControlFlowGraph::Use(ir::Node* op) const {
  auto it = uses_.find(op);
  PADDLE_ENFORCE(
      it != uses_.end(),
      string::Sprintf("Expect %s in live_out, but Not Found.", op->Name()));
  return it->second;
}

const std::vector<ir::Node*> ControlFlowGraph::Ops() const { return ops_; }

std::vector<ir::Node*>& ControlFlowGraph::Ops() { return ops_; }

ir::Node* ControlFlowGraph::GetNodeFromVarName(const std::string& name,
                                               ir::Node* op) const {
  // in ssa-graph, different version nodes have same name,
  // this function get the latest version var before target op
  // It may return nullptr, such as data node.
  ir::Node* found_node = nullptr;
  for (auto* node : ops_) {
    if (node == op) break;
    for (auto& output : node->outputs) {
      if (output->Name() == name) {
        found_node = output;
      }
    }
  }
  return found_node;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(analysis_var_pass, paddle::framework::details::AnalysisVarPass)
    .RequireGraphAttr(paddle::framework::details::kGraphNodePool)
    .RequireGraphAttr(paddle::framework::details::kAllOpDescs);
